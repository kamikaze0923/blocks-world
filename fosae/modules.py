import torch
from torch import nn
from fosae.gumble import gumbel_softmax, device
from fosae.activations import TrinaryStep

OBJS = 1
STACKS = 4
REMOVE_BG = True

P1 = 1
P2 = 1
ACTION_A = 3
CONV_CHANNELS = 16
ENCODER_FC_LAYER_SIZE = 100
PRED_BITS = 1
assert PRED_BITS == 1 or PRED_BITS == 2

IMG_H = 64
IMG_W = 96
IMG_C = 3
assert IMG_W % 4 == 0
assert IMG_H % 4 == 0
FMAP_H = IMG_H // 4
FMAP_W = IMG_W // 4

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

    def __init__(self, in_objects):
        super(PredicateNetwork, self).__init__()
        self.predicate_encoder = BaseObjectImageEncoder(in_objects=in_objects, out_features=PRED_BITS)

    def forward(self, input, temp):
        logits = self.predicate_encoder(input).view(-1, PRED_BITS)
        return gumbel_softmax(logits, temp)


class FoSae(nn.Module):

    def __init__(self):
        super(FoSae, self).__init__()
        self.unary_predicate_encoders = nn.ModuleList([PredicateNetwork(in_objects=1) for _ in range(P1)])
        self.binary_predicate_encoders = nn.ModuleList([PredicateNetwork(in_objects=2) for _ in range(P2)])
        # self.predicate_learner = PredicateLearner()
        # self.decoder = ObjectImageDecoder()

    def enumerate_state_singles_tuples(self, state, state_next, state_tilda, n_obj):
        pred_adjaceny = []
        state_adjaceny = []

        all_tuples = []
        all_tuples_next = []
        all_tuples_tilda = []
        all_singles = []
        all_singles_next = []
        all_singles_tilda = []
        state_adj_vec = torch.zeros(size=(state.size()[0],1))
        pred_adj_vec = torch.zeros(size=((n_obj **2).sum(), 1))
        obj_cnt = 0
        for i_state, (s, s_n, s_t, n) in enumerate(zip(state, state_next, state_tilda, n_obj)):
            for t in range(n):
                all_singles.append(s[t])
                all_singles_next.append(s_n[t])
                all_singles_tilda.append(s_t[t])
            n_pred = n.item() ** 2
            enum_index = torch.cartesian_prod(torch.arange(n.item()), torch.arange(n.item())).to(device)
            for t in enum_index:
                all_tuples.append(torch.index_select(s, dim=0, index=t).view(2 * IMG_C, IMG_H, IMG_W))
                all_tuples_next.append(torch.index_select(s_n, dim=0, index=t).view(2 * IMG_C, IMG_H, IMG_W))
                all_tuples_tilda.append(torch.index_select(s_t, dim=0, index=t).view(2 * IMG_C, IMG_H, IMG_W))
                state_adjaceny.append(torch.index_fill(state_adj_vec, dim=0, index=torch.tensor(i_state), value=1))
                pred_adjaceny.append(
                    torch.index_fill(pred_adj_vec, dim=0, index=torch.arange(start=obj_cnt, end=obj_cnt+n_pred), value=1)
                )
            obj_cnt += n_pred

        pred_adjaceny = torch.cat(pred_adjaceny, dim=1)
        state_adjaceny = torch.cat(state_adjaceny, dim=1)
        pred_adjaceny = pred_adjaceny / pred_adjaceny.sum(dim=1, keepdim=True)
        state_adjaceny = state_adjaceny / state_adjaceny.sum(dim=1, keepdim=True)

        return torch.stack(all_tuples, dim=0).to(device), torch.stack(all_tuples_next, dim=0).to(device), torch.stack(all_tuples_tilda, dim=0).to(device),\
               torch.stack(all_singles, dim=0).to(device), torch.stack(all_singles_next, dim=0).to(device), torch.stack(all_singles_tilda, dim=0).to(device),\
               pred_adjaceny.to(device), state_adjaceny.to(device)

    def forward(self, input, temp):
        state, state_next, state_tilda, n_obj = input

        obj_tuples, obj_tuples_next, obj_tuples_tilda, obj_singles, obj_singles_next, obj_singles_tilda, _, _ = \
            self.enumerate_state_singles_tuples(state, state_next, state_tilda, n_obj)


        binary_preds = torch.cat([pred_net(obj_tuples, temp) for pred_net in self.binary_predicate_encoders], dim=1)
        binary_preds_next = torch.cat([pred_net(obj_tuples_next, temp) for pred_net in self.binary_predicate_encoders], dim=1)
        binary_preds_tilda = torch.cat([pred_net(obj_tuples_tilda, temp) for pred_net in self.binary_predicate_encoders], dim=1)

        binary_preds_reshape = torch.zeros(size=(state.size()[0], n_obj.max()**2, P2)).to(device)
        binary_preds_next_reshape = torch.zeros(size=(state_next.size()[0], n_obj.max()**2, P2)).to(device)
        binary_preds_tilda_reshape = torch.zeros(size=(state_tilda.size()[0], n_obj.max()**2, P2)).to(device)

        start_idx = 0
        for i, n in enumerate(n_obj):
            binary_preds_reshape[i, :n**2, :] = binary_preds[start_idx: start_idx+n**2]
            binary_preds_next_reshape[i, :n**2, :] = binary_preds_next[start_idx: start_idx+n**2]
            binary_preds_tilda_reshape[i, :n**2, :] = binary_preds_tilda[start_idx: start_idx+n**2]
            start_idx += n**2

        unary_preds = torch.cat([pred_net(obj_singles, temp) for pred_net in self.unary_predicate_encoders], dim=1)
        unary_preds_next = torch.cat([pred_net(obj_singles_next, temp) for pred_net in self.unary_predicate_encoders], dim=1)
        unary_preds_tilda = torch.cat([pred_net(obj_singles_tilda, temp) for pred_net in self.unary_predicate_encoders], dim=1)

        unary_preds_reshape = torch.zeros(size=(state.size()[0], n_obj.max(), P1)).to(device)
        unary_preds_next_reshape = torch.zeros(size=(state_next.size()[0], n_obj.max(), P1)).to(device)
        unary_preds_tilda_reshape = torch.zeros(size=(state_tilda.size()[0], n_obj.max(), P1)).to(device)

        start_idx = 0
        for i, n in enumerate(n_obj):
            unary_preds_reshape[i, :n, :] = unary_preds[start_idx: start_idx+n]
            unary_preds_next_reshape[i, :n, :] = unary_preds_next[start_idx: start_idx+n]
            unary_preds_tilda_reshape[i, :n, :] = unary_preds_tilda[start_idx: start_idx+n]
            start_idx += n

        return binary_preds_reshape, binary_preds_next_reshape, binary_preds_tilda_reshape, \
               unary_preds_reshape, unary_preds_next_reshape, unary_preds_tilda_reshape


class ActionEncoder(nn.Module):

    def __init__(self):
        super(ActionEncoder, self).__init__()
        self.state_action_encoder = BaseObjectImageEncoder(in_objects=2+ACTION_A, out_features=PRED_BITS)
        self.step_func = TrinaryStep()

    def forward(self, input, temp):
        state, action, n_obj = input
        obj_action_tuples = self.enumerate_state_action_tuples(state, action, n_obj)
        logits = self.state_action_encoder(obj_action_tuples)
        logits = logits.view(-1, 1)
        # probs = gumbel_softmax(logits, temp)
        # target = torch.tensor([-1, 0, 1]).expand_as(probs).to(device)
        # change = torch.mul(probs, target).sum(dim=-1, keepdim=True)
        return self.step_func.apply(logits)

    def enumerate_state_action_tuples(self, state, action, n_obj):
        all_tuples = []
        for s, a, n in zip(state, action, n_obj):
            a = a.view(ACTION_A * IMG_C, IMG_H, IMG_W)
            enum_index = torch.cartesian_prod(torch.arange(n.item()), torch.arange(n.item())).to(device)
            for t in enum_index:
                objs = torch.index_select(s, dim=0, index=t).view(2*IMG_C, IMG_H, IMG_W)
                all_tuples.append(torch.cat([objs, a], dim=0))
        return torch.stack(all_tuples, dim=0)


class FoSae_Action(nn.Module):

    def __init__(self):
        super(FoSae_Action, self).__init__()
        self.action_models = nn.ModuleList([ActionEncoder() for _ in range(P2)])

    def forward(self, input, temp):
        state, action, n_obj = input

        all_changes = []

        for model in self.action_models:
            preds_change = model(input, temp)
            all_changes.append(preds_change)

        all_changes = torch.cat(all_changes, dim=1)

        changes_reshape = torch.zeros(size=(state.size()[0], n_obj.max()**2, P)).to(device)

        start_idx = 0
        for i, n in enumerate(n_obj):
            changes_reshape[i, :n**2, :] = all_changes[start_idx: start_idx+n**2]
            start_idx += n**2

        return changes_reshape