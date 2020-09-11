import torch
from torch import nn
from fosae.gumble import gumbel_softmax, device
from fosae.activations import TrinaryStep

OBJS = 1
STACKS = 4
REMOVE_BG = True

N = OBJS + STACKS + (0 if REMOVE_BG else 1)
P = 2
A = 2
U = 1
CONV_CHANNELS = 16
ENCODER_FC_LAYER_SIZE = 200
DECODER_FC_LAYER_SIZE = 2000
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
        self.predicate_encoder = BaseObjectImageEncoder(in_objects=A, out_features=PRED_BITS)

    def forward(self, input, temp):
        logits = self.predicate_encoder(input).view(-1, N**A, PRED_BITS)
        return gumbel_softmax(logits, temp)


class PredicateUnit(nn.Module):

    def __init__(self, predicate_nets, n_obj):
        super(PredicateUnit, self).__init__()
        self.predicate_nets = predicate_nets
        self.enum_index = torch.cartesian_prod(torch.arange(n_obj), torch.arange(n_obj)).to(device)

    def forward(self, state, state_next, temp):

        obj_tuples = self.enumerate_state_tuples(state)
        preds = torch.stack([pred_net(obj_tuples, temp) for pred_net in self.predicate_nets], dim=1)

        obj_tuples_next = self.enumerate_state_tuples(state_next)
        preds_next = torch.stack([pred_net(obj_tuples_next, temp) for pred_net in self.predicate_nets], dim=1)

        return preds, preds_next

    def enumerate_state_tuples(self, state):
        all_tuples = []
        for t in self.enum_index:
            all_tuples.append(torch.index_select(state, dim=1, index=t).view(-1, A*IMG_C, IMG_H, IMG_W))
        all_tuples = torch.stack(all_tuples, dim=1)
        return all_tuples

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
        self.predicate_units = nn.ModuleList([PredicateUnit(self.predicate_nets, N) for _ in range(U)])
        self.decoder = PredicateDecoder()

    def forward(self, input, temp):
        state, state_next = input

        all_preds = []
        all_preds_next = []

        for pu in self.predicate_units:
            preds, preds_next = pu(state, state_next, temp)

            all_preds.append(preds)
            all_preds_next.append(preds_next)

        all_preds, _ = torch.stack(all_preds, dim=1)[:,:,:,:,[0]].max(dim=1)
        all_preds_next, _ = torch.stack(all_preds_next, dim=1)[:,:,:,:,[0]].max(dim=1)

        x_hat = self.decoder(all_preds)
        x_hat_next = self.decoder(all_preds_next)

        return (x_hat, x_hat_next), (all_preds, all_preds_next)


class ActionEncoder(nn.Module):

    def __init__(self, n_obj):
        super(ActionEncoder, self).__init__()
        self.state_action_encoder = BaseObjectImageEncoder(in_objects=A+ACTION_A, out_features=1)
        self.enum_index = torch.cartesian_prod(torch.arange(n_obj), torch.arange(n_obj)).to(device)
        self.step_func = TrinaryStep()

    def forward(self, input, temp):
        state, action = input
        obj_action_tuples = self.enumerate_state_action_tuples(state, action)
        logits = self.state_action_encoder(obj_action_tuples)
        logits = logits.view(-1, N**A, 1)
        # probs = gumbel_softmax(logits, temp)
        # target = torch.tensor([-1, 0, 1]).expand_as(probs).to(device)
        # change = torch.mul(probs, target).sum(dim=-1, keepdim=True)
        return self.step_func.apply(logits)

    def enumerate_state_action_tuples(self, state, action):
        action = action.view(-1, A*IMG_C, IMG_H, IMG_W)
        all_tuples = []
        for t in self.enum_index:
            objs = torch.index_select(state, dim=1, index=t).view(-1, A*IMG_C, IMG_H, IMG_W)
            all_tuples.append(torch.cat([objs, action], dim=1))
        all_tuples = torch.stack(all_tuples, dim=1)
        return all_tuples


class FoSae_Action(nn.Module):

    def __init__(self):
        super(FoSae_Action, self).__init__()
        self.action_models = nn.ModuleList([ActionEncoder(N) for _ in range(P)])

    def forward(self, input, temp):

        all_changes = []

        for model in self.action_models:
            preds_change = model(input, temp)
            all_changes.append(preds_change)

        all_changes = torch.stack(all_changes, dim=1)

        return all_changes