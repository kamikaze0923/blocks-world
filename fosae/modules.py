import torch
from torch import nn
from fosae.gumble import gumbel_softmax, device

N = 9
P = 36
A = 2
U = 18
CONV_CHANNELS = 32
FC_LAYER_SIZE = 1000

IMG_H = 64
IMG_W = 96
assert IMG_W % 4 == 0
assert IMG_H % 4 == 0
FMAP_H = IMG_H //4
FMAP_W = IMG_W //4
IMG_C = 3
ACTION_A = 2

class BackBoneImageObjectEncoder(nn.Module):

    def __init__(self, in_objects, out_features):
        super(BackBoneImageObjectEncoder, self).__init__()
        self.in_objects = in_objects
        self.conv1 = nn.Conv2d(in_channels=in_objects*IMG_C, out_channels=CONV_CHANNELS, kernel_size=(4,4), stride=(2,2), padding=1)
        self.bn1 = nn.BatchNorm2d(CONV_CHANNELS)
        self.dpt1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv2d(in_channels=CONV_CHANNELS, out_channels=CONV_CHANNELS, kernel_size=(2,2), stride=(2,2))
        self.bn2 = nn.BatchNorm2d(CONV_CHANNELS)
        self.dpt2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(in_features=CONV_CHANNELS*FMAP_H*FMAP_W, out_features=FC_LAYER_SIZE)
        self.bn3 = nn.BatchNorm1d(1)
        self.dpt3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(in_features=FC_LAYER_SIZE, out_features=out_features)

    def forward(self, input, temp):
        h1 = self.dpt1(self.bn1(self.conv1(input.view(-1, self.in_objects * IMG_C, IMG_H, IMG_W))))
        h2 = self.dpt2(self.bn2(self.conv2(h1)))
        h2 = h2.view(-1, 1, CONV_CHANNELS * FMAP_H * FMAP_W)
        h3 = self.dpt3(self.bn3(self.fc3(h2)))
        return self.fc4(h3)


class PredicateNetwork(nn.Module):

    def __init__(self):
        super(PredicateNetwork, self).__init__()
        self.predicate_encoder = BackBoneImageObjectEncoder(in_objects=A, out_features=P*2)

    def forward(self, input, temp):
        logits = self.predicate_encoder(input, temp)
        logits = logits.view(-1, P, 2)
        prob = gumbel_softmax(logits, temp)
        return prob

class StateEncoder(nn.Module):

    def __init__(self):
        super(StateEncoder, self).__init__()
        self.objects_encoder = BackBoneImageObjectEncoder(in_objects=N, out_features=A*N)

    def forward(self, input, temp):
        logits = self.objects_encoder(input, temp)
        logits = logits.view(-1, A, N)
        prob = gumbel_softmax(logits, temp).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, A, -1, IMG_C, IMG_H, IMG_W)
        dot = torch.mul(prob, input.unsqueeze(1).expand(-1, A, -1 ,-1, -1, -1)).sum(dim=2)
        return dot

class ActionEncoder(nn.Module):

    def __init__(self):
        super(ActionEncoder, self).__init__()
        self.state_action_encoder = BackBoneImageObjectEncoder(in_objects=N+ACTION_A, out_features=P*3)

    def forward(self, input, temp, action_base=torch.tensor([-1.0, 0.0, 1.0])):
        logits = self.state_action_encoder(input, temp)
        logits = logits.view(-1, P, 3)
        action_one_hot = gumbel_softmax(logits, temp)
        action_base_expand = action_base.expand_as(action_one_hot).to(device)
        action = torch.mul(action_one_hot, action_base_expand).sum(dim=-1, keepdim=True)
        return torch.cat([action, -action], dim=-1)

class PredicateUnit(nn.Module):

    def __init__(self, predicate_net):
        super(PredicateUnit, self).__init__()
        self.state_encoder = StateEncoder().to(device)
        self.action_encoder = ActionEncoder().to(device)
        self.predicate_net = predicate_net

    def forward(self, input, temp):
        state, state_next, action = input

        args = self.state_encoder(state, temp)
        preds = self.predicate_net(args, temp)

        args_next = self.state_encoder(state, temp)
        preds_next = self.predicate_net(args, temp)

        action = self.action_encoder(torch.cat([state, action], dim=1), temp)

        return args, args_next, preds, preds_next, action

class PredicateDecoder(nn.Module):

    def __init__(self):
        super(PredicateDecoder, self).__init__()
        self.fc1 = nn.Linear(in_features=U*P*2, out_features=FC_LAYER_SIZE)
        self.bn1 = nn.BatchNorm1d(1)
        self.dpt1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=FC_LAYER_SIZE, out_features=FC_LAYER_SIZE)
        self.bn2 = nn.BatchNorm1d(1)
        self.dpt2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(in_features=FC_LAYER_SIZE, out_features=N*IMG_C*IMG_H*IMG_W)

    def forward(self, input):
        h1 = self.dpt1(self.bn1(self.fc1(input.view(-1, 1, U*P*2))))
        h2 = self.dpt2(self.bn2(self.fc2(h1)))
        return torch.sigmoid(self.fc3(h2)).view(-1, N, IMG_C, IMG_H, IMG_W)

class FoSae(nn.Module):

    def __init__(self):
        super(FoSae, self).__init__()
        self.predicate_net = PredicateNetwork().to(device)
        self.pus = []
        for _ in range(U):
            self.pus.append(PredicateUnit(self.predicate_net))
        self.decoder = PredicateDecoder().to(device)

    def forward(self, x, temp):

        all_args = []
        all_args_next = []
        all_preds = []
        all_preds_next = []
        all_actions = []

        for pu in self.pus:
            args, args_next, preds, preds_next, actions = pu(x, temp)
            all_args.append(args)
            all_args_next.append(args_next)
            all_preds.append(preds)
            all_preds_next.append(preds_next)
            all_actions.append(actions)

        all_args = torch.stack(all_args, dim=1)
        all_args_next = torch.stack(all_args_next, dim=1)
        all_preds = torch.stack(all_preds, dim=1)
        all_preds_next = torch.stack(all_preds_next, dim=1)
        all_actions = torch.stack(all_actions, dim=1)

        x_hat = self.decoder(all_preds)
        x_hat_next = self.decoder(all_preds_next)

        return (x_hat, x_hat_next), (all_args, all_args_next), (all_preds, all_preds_next, all_actions)
