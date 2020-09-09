import torch
from torch import nn
from fosae.gumble import gumbel_softmax, device
from fosae.activations import TrinaryStep

OBJS = 2
STACKS = 4
REMOVE_BG = True

N = OBJS + STACKS + (0 if REMOVE_BG else 1)
P = 6
A = 2
U = 6
CONV_CHANNELS = 16
ENCODER_FC_LAYER_SIZE = 200
DECODER_FC_LAYER_SIZE = 1000
PRED_BITS = 1
assert PRED_BITS == 1 or PRED_BITS == 2

IMG_H = 64
IMG_W = 96
assert IMG_W % 4 == 0
assert IMG_H % 4 == 0
FMAP_H = IMG_H //4
FMAP_W = IMG_W //4
IMG_C = 3
ACTION_A = 2

class BaseObjectImageEncoder(nn.Module):

    def __init__(self, in_objects, out_features):
        super(BaseObjectImageEncoder, self).__init__()
        self.in_objects = in_objects
        self.conv1 = nn.Conv2d(in_channels=in_objects*IMG_C, out_channels=CONV_CHANNELS, kernel_size=(8,8), stride=(4,4), padding=2)
        # self.bn1 = nn.BatchNorm2d(CONV_CHANNELS)
        self.fc2 = nn.Linear(in_features=CONV_CHANNELS*FMAP_H*FMAP_W, out_features=ENCODER_FC_LAYER_SIZE)
        # self.bn2 = nn.BatchNorm1d(1)
        self.fc3 = nn.Linear(in_features=ENCODER_FC_LAYER_SIZE, out_features=out_features)

    def forward(self, input):
        h1 = torch.relu(self.conv1(input.view(-1, self.in_objects * IMG_C, IMG_H, IMG_W)))
        h1 = h1.view(-1, 1, CONV_CHANNELS * FMAP_H * FMAP_W)
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2)

class PredicateNetwork(nn.Module):

    def __init__(self):
        super(PredicateNetwork, self).__init__()
        self.predicate_encoder = BaseObjectImageEncoder(in_objects=A, out_features=N**A*PRED_BITS)

    def forward(self, input, prob, temp):
        prod = prob[:, 0, :]
        for i in range(1, A):
            prod = prod.unsqueeze(-1)
            prod = torch.bmm(prod, prob[:, i, :].unsqueeze(1)).flatten(start_dim=1)
        logits = self.predicate_encoder(input).view(-1, N**A, PRED_BITS)
        return torch.mul(gumbel_softmax(logits, temp), prod.unsqueeze(-1).expand(-1, -1, PRED_BITS))

class StateEncoder(nn.Module):

    def __init__(self):
        super(StateEncoder, self).__init__()
        self.objects_encoder = BaseObjectImageEncoder(in_objects=N, out_features=A*N)

    def forward(self, input, temp):
        logits = self.objects_encoder(input)
        logits = logits.view(-1, A, N)
        prob = gumbel_softmax(logits, temp)
        dot = torch.mul(
            prob.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, A, N, IMG_C, IMG_H, IMG_W),
            input.unsqueeze(1).expand(-1, A, -1 ,-1, -1, -1)
        ).sum(dim=2)
        return dot, prob

class ActionEncoder(nn.Module):

    def __init__(self):
        super(ActionEncoder, self).__init__()
        self.state_action_encoder = BaseObjectImageEncoder(in_objects=N+ACTION_A, out_features=N**A)
        self.step_func = TrinaryStep()

    def forward(self, input):
        logits = self.state_action_encoder(input)
        logits = logits.view(-1, N**A, 1)
        return self.step_func.apply(logits)

class PredicateUnit(nn.Module):

    def __init__(self, predicate_nets):
        super(PredicateUnit, self).__init__()
        self.state_encoder = StateEncoder()
        self.predicate_nets = predicate_nets

    def forward(self, state, state_next, temp):

        args, probs = self.state_encoder(state, temp)
        preds = torch.stack([pred_net(args, probs, temp) for pred_net in self.predicate_nets], dim=1)

        args_next, probs_next = self.state_encoder(state_next, temp)
        preds_next = torch.stack([pred_net(args_next, probs_next, temp) for pred_net in self.predicate_nets], dim=1)

        return args, args_next, preds, preds_next

class PredicateDecoder(nn.Module):

    def __init__(self):
        super(PredicateDecoder, self).__init__()
        self.fc1 = nn.Linear(in_features=P*N**A, out_features=DECODER_FC_LAYER_SIZE)
        # self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(in_features=DECODER_FC_LAYER_SIZE, out_features=N*IMG_C*IMG_H*IMG_W)

    def forward(self, input):
        h1 = torch.relu(self.fc1(input.view(-1, 1, P*N**A)))
        return torch.sigmoid(self.fc2(h1)).view(-1, N, IMG_C, IMG_H, IMG_W)

class FoSae(nn.Module):

    def __init__(self):
        super(FoSae, self).__init__()
        self.predicate_nets = nn.ModuleList([PredicateNetwork() for _ in range(P)])
        self.action_encoders = nn.ModuleList([ActionEncoder() for _ in range(P)])
        self.predicate_units = nn.ModuleList([PredicateUnit(self.predicate_nets) for _ in range(U)])
        self.decoder = PredicateDecoder()

    def forward(self, input, temp):
        state, state_next, action = input

        all_args = []
        all_args_next = []
        all_preds = []
        all_preds_next = []

        for pu in self.predicate_units:
            args, args_next, preds, preds_next = pu(state, state_next, temp)
            all_args.append(args)
            all_args_next.append(args_next)
            all_preds.append(preds)
            all_preds_next.append(preds_next)

        all_args = torch.stack(all_args, dim=1)
        all_args_next = torch.stack(all_args_next, dim=1)
        all_preds, _ = torch.stack(all_preds, dim=1)[:,:,:,:,[0]].max(dim=1)
        all_preds_next, _ = torch.stack(all_preds_next, dim=1)[:,:,:,:,[0]].max(dim=1)

        all_actions = torch.stack([act_net(torch.cat([state, action], dim=1)) for act_net in self.action_encoders], dim=1)

        all_preds_next_by_action = all_preds.detach() + all_actions

        x_hat = self.decoder(all_preds)
        x_hat_next = self.decoder(all_preds_next)
        x_hat_next_by_action = self.decoder(all_preds_next_by_action)

        return (x_hat, x_hat_next, x_hat_next_by_action), (all_args, all_args_next), (all_preds, all_preds_next, all_preds_next_by_action)