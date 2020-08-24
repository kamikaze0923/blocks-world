import torch
from torch import nn
from fosae.gumble import gumbel_softmax

N = 9
P = 9
A = 1
U = 9
LAYER_SIZE = 300
N_OBJ_FEATURE = 50 * 75

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
        self.decoder = PredicateDecoder()


    def encode(self, x):
        return

    def decode(self, z_y):
        return

    def forward(self, x, temp):
        x_u = x.unsqueeze(1).expand(-1, U, -1, -1) #copy x for multiple predicate units
        args = self.encoder(x_u, temp)
        preds = self.predicate_net(args, temp)
        out = self.decoder(preds)
        return out, args, preds



