import torch
from torch import nn
from fosae.gumble import gumbel_softmax, device
import itertools

N = 9
P = 9
A = 2
U = 9
CONV_CHANNELS = 16
ENCODER_FC_LAYER_SIZE = 100
DECODER_FC_LAYER_SIZE = 1000

IMG_H = 64
IMG_W = 96
assert IMG_W % 4 == 0
assert IMG_H % 4 == 0
FMAP_H = IMG_H //4
FMAP_W = IMG_W //4
IMG_C = 3
ACTION_A = 2

class BaseObjectImageEncoder(nn.Module):

    def __init__(self, in_objects, out_features, fc_size=ENCODER_FC_LAYER_SIZE):
        super(BaseObjectImageEncoder, self).__init__()
        self.in_objects = in_objects
        self.conv1 = nn.Conv2d(in_channels=in_objects*IMG_C, out_channels=CONV_CHANNELS, kernel_size=(8,8), stride=(4,4), padding=2)
        self.bn1 = nn.BatchNorm2d(CONV_CHANNELS)
        self.fc2 = nn.Linear(in_features=CONV_CHANNELS*FMAP_H*FMAP_W, out_features=fc_size)
        self.bn2 = nn.BatchNorm1d(1)
        self.fc3 = nn.Linear(in_features=fc_size, out_features=out_features)

    def forward(self, input):
        h1 = self.bn1(torch.relu(self.conv1(input.view(-1, self.in_objects * IMG_C, IMG_H, IMG_W))))
        h1 = h1.view(-1, 1, CONV_CHANNELS * FMAP_H * FMAP_W)
        h2 = self.bn2(torch.relu(self.fc2(h1)))
        return self.fc3(h2)

class PredicateNetwork(nn.Module):

    def __init__(self):
        super(PredicateNetwork, self).__init__()
        self.predicate_encoder = BaseObjectImageEncoder(in_objects=A, out_features=2)

    def forward(self, input):
        logits = self.predicate_encoder(input)
        return logits.view(-1, 2)


class StateEncoder(nn.Module):

    def __init__(self):
        super(StateEncoder, self).__init__()
        self.objects_encoder = BaseObjectImageEncoder(in_objects=N, out_features=A*N)

    def forward(self, input, temp):
        logits = self.objects_encoder(input)
        logits = logits.view(-1, A, N)
        prob = gumbel_softmax(logits, temp).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, A, -1, IMG_C, IMG_H, IMG_W)
        dot = torch.mul(prob, input.unsqueeze(1).expand(-1, A, -1 ,-1, -1, -1)).sum(dim=2)
        return dot

class ActionEncoder(nn.Module):

    def __init__(self):
        super(ActionEncoder, self).__init__()
        self.state_action_encoder = BaseObjectImageEncoder(in_objects=N+ACTION_A, out_features=2, fc_size=ENCODER_FC_LAYER_SIZE*10)

    def forward(self, input):
        logits = self.state_action_encoder(input)
        return logits.view(-1, 2)


class PredicateUnit(nn.Module):

    def __init__(self, predicate_nets, action_encoders):
        super(PredicateUnit, self).__init__()
        self.state_encoder = StateEncoder()
        self.action_encoders = action_encoders
        self.predicate_nets = predicate_nets

    def forward(self, input, temp):
        state, state_next, action = input

        args = self.state_encoder(state, temp)
        preds = torch.stack([pred_net(args) for pred_net in self.predicate_nets], dim=1)

        args_next = self.state_encoder(state_next, temp)
        preds_next = torch.stack([pred_net(args_next) for pred_net in self.predicate_nets], dim=1)

        action_latent = torch.stack([act_net(torch.cat([state, action], dim=1)) for act_net in self.action_encoders], dim=1)

        return args, args_next, gumbel_softmax(preds, temp), gumbel_softmax(preds_next, temp), gumbel_softmax(preds.detach() + action_latent, temp)



class PredicateDecoder(nn.Module):

    def __init__(self):
        super(PredicateDecoder, self).__init__()
        self.fc1 = nn.Linear(in_features=U*P*2, out_features=DECODER_FC_LAYER_SIZE)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(in_features=DECODER_FC_LAYER_SIZE, out_features=N*IMG_C*IMG_H*IMG_W)

    def forward(self, input):
        h1 = self.bn1(torch.relu(self.fc1(input.view(-1, 1, U*P*2))))
        return torch.sigmoid(self.fc2(h1)).view(-1, N, IMG_C, IMG_H, IMG_W)

class FoSae(nn.Module):

    def __init__(self):
        super(FoSae, self).__init__()
        self.predicate_nets = nn.ModuleList([PredicateNetwork() for _ in range(P)])
        self.action_encoders = nn.ModuleList([ActionEncoder() for _ in range(P)])
        self.predicate_units = nn.ModuleList([PredicateUnit(self.predicate_nets, self.action_encoders) for _ in range(U)])
        self.decoder = PredicateDecoder()

    def forward(self, x, temp):

        all_args = []
        all_args_next = []
        all_preds = []
        all_preds_next = []
        all_preds_next_by_action = []

        for pu in self.predicate_units:
            args, args_next, preds, preds_next, preds_next_by_action = pu(x, temp)
            all_args.append(args)
            all_args_next.append(args_next)
            all_preds.append(preds)
            all_preds_next.append(preds_next)
            all_preds_next_by_action.append(preds_next_by_action)

        all_args = torch.stack(all_args, dim=1)
        all_args_next = torch.stack(all_args_next, dim=1)
        all_preds = torch.stack(all_preds, dim=1)
        all_preds_next = torch.stack(all_preds_next, dim=1)
        all_preds_next_by_action = torch.stack(all_preds_next_by_action, dim=1)

        x_hat = self.decoder(all_preds)
        x_hat_next = self.decoder(all_preds_next)
        x_hat_next_by_action = self.decoder(all_preds_next_by_action)

        return (x_hat, x_hat_next, x_hat_next_by_action), (all_args, all_args_next), (all_preds, all_preds_next, all_preds_next_by_action)
