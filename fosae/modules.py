import torch
from torch import nn
from fosae.gumble import gumbel_softmax

N = 9
P = 2
A = 2
U = 9
LAYER_SIZE = 100
IMG_H = 50
IMG_W = 75
IMG_C = 3
N_OBJ_FEATURE = IMG_H * IMG_W * IMG_C
ACTION_A = 2

class AttentionEncoder(nn.Module):

    def __init__(self):
        super(AttentionEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=U, out_channels=U*LAYER_SIZE, kernel_size=N*N_OBJ_FEATURE)
        self.bn1 = nn.BatchNorm1d(U*LAYER_SIZE)
        self.dpt1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(in_channels=U, out_channels=U*A*N, kernel_size=LAYER_SIZE)

    def forward(self, input, temp):
        h1 = self.dpt1(self.bn1(self.conv1(input.view(-1, U, N*N_OBJ_FEATURE)))).view(-1, U, LAYER_SIZE)
        logits = self.conv2(h1).view(-1, U, A, N)
        prob = gumbel_softmax(logits, temp).unsqueeze(-1).expand(-1, -1, -1, -1, N_OBJ_FEATURE)
        dot = torch.mul(prob, input.unsqueeze(2).expand(-1, -1, A, -1 ,-1)).sum(dim=3)
        return dot

class PredicateNetwork(nn.Module):

    def __init__(self):
        super(PredicateNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=A*N_OBJ_FEATURE, out_features=LAYER_SIZE)
        self.bn1 = nn.BatchNorm1d(U)
        self.dpt1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=LAYER_SIZE, out_features=P*2)

    def forward(self, input, temp):
        h1 = self.dpt1(self.bn1(self.fc1(input.view(-1, U, A*N_OBJ_FEATURE))))
        logits = self.fc2(h1).view(-1, U, P, 2)
        prob = gumbel_softmax(logits, temp)
        return prob

class ActionNetwork(nn.Module):

    def __init__(self):
        super(ActionNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=(ACTION_A+A)*N_OBJ_FEATURE, out_features=LAYER_SIZE)
        self.bn1 = nn.BatchNorm1d(U)
        self.dpt1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=LAYER_SIZE, out_features=P*2)
        self.hardTH = nn.Hardtanh(-3, 3)

    def forward(self, input):
        h1 = self.dpt1(self.bn1(self.fc1(input.view(-1, U, (ACTION_A+A)*N_OBJ_FEATURE))))
        logits = self.fc2(h1).view(-1, U, P, 2)
        action = self.hardTH(logits)
        return action


class PredicateDecoder(nn.Module):

    def __init__(self):
        super(PredicateDecoder, self).__init__()
        self.fc1 = nn.Linear(in_features=U*P*2, out_features=LAYER_SIZE)
        self.bn1 = nn.BatchNorm1d(1)
        self.dpt1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=LAYER_SIZE, out_features=N*N_OBJ_FEATURE)

    def forward(self, input):
        h1 = self.dpt1(self.bn1(self.fc1(input.view(-1, 1, U*P*2))))
        return torch.sigmoid(self.fc2(h1)).view(-1, N, N_OBJ_FEATURE)


class FoSae(nn.Module):

    def __init__(self):
        super(FoSae, self).__init__()
        self.encoder = AttentionEncoder()
        self.predicate_net = PredicateNetwork()
        self.action_net = ActionNetwork()
        self.decoder = PredicateDecoder()

    def forward(self, x, temp):
        x_pre, x_next, x_action = x

        args_pre = self.encoder(x_pre.unsqueeze(1).expand(-1, U, -1, -1), temp) #copy x for multiple predicate units
        preds_pre = self.predicate_net(args_pre, temp)
        out_pre = self.decoder(preds_pre)

        state_action = torch.cat([args_pre, x_action.unsqueeze(1).expand(-1, U, -1, -1)], dim=2)
        preds_action = self.action_net(state_action)

        args_next = self.encoder(x_next.unsqueeze(1).expand(-1, U, -1, -1), temp) #copy x for multiple predicate units
        preds_next = self.predicate_net(args_next, temp)
        out_next = self.decoder(preds_next)

        return (out_pre, out_next), (args_pre, args_next), (preds_pre, preds_next, preds_action)
