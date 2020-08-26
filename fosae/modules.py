import torch
from torch import nn
from fosae.gumble import gumbel_softmax, device

N = 9
P = 32
A = 2
U = 9
LAYER_SIZE = 200
IMG_H = 50
IMG_W = 75
IMG_C = 3
N_OBJ_FEATURE = IMG_H * IMG_W * IMG_C
ACTION_A = 2

class PredicateNetwork(nn.Module):

    def __init__(self):
        super(PredicateNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=A*N_OBJ_FEATURE, out_features=LAYER_SIZE)
        self.bn1 = nn.BatchNorm1d(1)
        self.dpt1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=LAYER_SIZE, out_features=P*2)

    def forward(self, input, temp):
        h1 = self.dpt1(self.bn1(self.fc1(input.view(-1, 1, A*N_OBJ_FEATURE))))
        logits = self.fc2(h1).view(-1, P, 2)
        prob = gumbel_softmax(logits, temp)
        return prob

class StateEncoder(nn.Module):

    def __init__(self):
        super(StateEncoder, self).__init__()
        self.fc1 = nn.Linear(in_features=N*N_OBJ_FEATURE, out_features=LAYER_SIZE)
        self.bn1 = nn.BatchNorm1d(1)
        self.dpt1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=LAYER_SIZE, out_features=A*N)

    def forward(self, input, temp):
        h1 = self.dpt1(self.bn1(self.fc1(input.view(-1, 1, N*N_OBJ_FEATURE))))
        logits = self.fc2(h1)
        logits = logits.view(-1, A, N)
        prob = gumbel_softmax(logits, temp).unsqueeze(-1).expand(-1, -1, -1, N_OBJ_FEATURE)
        dot = torch.mul(prob, input.unsqueeze(1).expand(-1, A, -1 ,-1)).sum(dim=2)
        return dot

class ActionEncoder(nn.Module):

    def __init__(self):
        super(ActionEncoder, self).__init__()
        self.fc1 = nn.Linear(in_features=(N+ACTION_A)*N_OBJ_FEATURE, out_features=LAYER_SIZE)
        self.bn1 = nn.BatchNorm1d(1)
        self.dpt1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=LAYER_SIZE, out_features=P*3)

    def forward(self, input, temp, action_base=torch.tensor([-1.0, 0.0, 1.0])):
        h1 = self.dpt1(self.bn1(self.fc1(input.view(-1, 1, (N+ACTION_A)*N_OBJ_FEATURE))))
        logits = self.fc2(h1)
        logits = logits.view(-1, P, 3)
        action_one_hot = gumbel_softmax(logits, temp)
        action_base_expand = action_base.expand_as(action_one_hot).to(device)
        action = torch.mul(action_one_hot, action_base_expand).sum(dim=-1, keepdim=True)
        return torch.cat([action, -action], dim=-1)

class PredicateUnit(nn.Module):

    def __init__(self, predicate_net):
        super(PredicateUnit, self).__init__()
        self.state_encoder = StateEncoder()
        self.action_encoder = ActionEncoder()
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
        self.predicate_net = PredicateNetwork()
        self.pus = []
        for _ in range(U):
            self.pus.append(PredicateUnit(self.predicate_net))
        self.decoder = PredicateDecoder()

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
