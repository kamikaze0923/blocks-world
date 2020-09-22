import torch
from torch import nn
from fosae.gumble import gumbel_softmax, device
from fosae.activations import TrinaryStep
from torch.autograd import Variable

OBJS = 1
STACKS = 4
REMOVE_BG = False

P = 1
A = 2
ACTION_A = 3
CONV_CHANNELS = 16
OBJECT_LATENT = 16
ENCODER_FC_LAYER_SIZE = 100
DECODER_FC_LAYER_SIZE = 1000
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

    def __init__(self):
        super(PredicateNetwork, self).__init__()
        self.predicate_encoder = BaseObjectImageEncoder(in_objects=A, out_features=PRED_BITS)

    def forward(self, input, temp):
        logits = self.predicate_encoder(input).view(-1, PRED_BITS)
        return gumbel_softmax(logits, temp)


class PredicateLearner(nn.Module):

    def __init__(self):
        super(PredicateLearner, self).__init__()
        self.fc1 = nn.Linear(in_features=P, out_features=OBJECT_LATENT, bias=False)
        self.fc2 = nn.Linear(in_features=OBJECT_LATENT, out_features=OBJECT_LATENT, bias=False)

    def forward(self, x, adj):
        a1 = torch.relu(torch.matmul(adj, self.fc1(x)))
        a2 = torch.relu(torch.matmul(adj, self.fc2(a1)))
        return a2

class ObjectImageDecoder(nn.Module):

    def __init__(self):
        super(ObjectImageDecoder, self).__init__()
        self.fc1 = nn.Linear(in_features=OBJECT_LATENT, out_features=DECODER_FC_LAYER_SIZE)
        self.fc2 = nn.Linear(in_features=DECODER_FC_LAYER_SIZE, out_features=IMG_C*IMG_H*IMG_W)

    def forward(self, input, state_adj_matrix):
        h1 = torch.relu(self.fc1(torch.matmul(state_adj_matrix, input)))
        return torch.sigmoid(self.fc2(h1)).view(-1, IMG_C, IMG_H, IMG_W)

class FoSae(nn.Module):

    def __init__(self):
        super(FoSae, self).__init__()
        self.predicate_encoders = nn.ModuleList([PredicateNetwork() for _ in range(P)])
        # self.predicate_learner = PredicateLearner()
        # self.decoder = ObjectImageDecoder()

    def enumerate_state_tuples(self, state, state_next, state_tilda, n_obj):
        pred_adjaceny = []
        state_adjaceny = []

        all_tuples = []
        all_tuples_next = []
        all_tuples_tilda = []
        state_adj_vec = torch.zeros(
            size=(state.size()[0],1)
        )
        pred_adj_vec = torch.zeros(
            size=((n_obj **2).sum(), 1)
        )
        obj_cnt = 0
        for i_state, (s, s_n, s_t, n) in enumerate(zip(state, state_next, state_tilda, n_obj)):
            n_pred = n.item() ** 2
            enum_index = torch.cartesian_prod(torch.arange(n.item()), torch.arange(n.item())).to(device)
            for t in enum_index:
                all_tuples.append(torch.index_select(s, dim=0, index=t).view(A * IMG_C, IMG_H, IMG_W))
                all_tuples_next.append(torch.index_select(s_n, dim=0, index=t).view(A * IMG_C, IMG_H, IMG_W))
                all_tuples_tilda.append(torch.index_select(s_t, dim=0, index=t).view(A * IMG_C, IMG_H, IMG_W))
                state_adjaceny.append(torch.index_fill(state_adj_vec, dim=0, index=torch.tensor(i_state), value=1))
                pred_adjaceny.append(
                    torch.index_fill(pred_adj_vec, dim=0, index=torch.arange(start=obj_cnt, end=obj_cnt+n_pred), value=1)
                )
            obj_cnt += n_pred

        pred_adjaceny = torch.cat(pred_adjaceny, dim=1).to(device)
        state_adjaceny = torch.cat(state_adjaceny, dim=1).to(device)
        pred_adjaceny = pred_adjaceny / pred_adjaceny.sum(dim=1, keepdim=True)
        state_adjaceny = state_adjaceny / state_adjaceny.sum(dim=1, keepdim=True)

        return torch.stack(all_tuples, dim=0), torch.stack(all_tuples_next, dim=0), torch.stack(all_tuples_tilda, dim=0), \
               pred_adjaceny, state_adjaceny

    def forward(self, input, temp):
        state, state_next, state_tilda, n_obj = input

        obj_tuples, obj_tuples_next, obj_tuples_tilda, pred_adj, state_adj = \
            self.enumerate_state_tuples(state, state_next, state_tilda, n_obj)

        preds = torch.cat([pred_net(obj_tuples, temp) for pred_net in self.predicate_encoders], dim=1)
        preds_next = torch.cat([pred_net(obj_tuples_next, temp) for pred_net in self.predicate_encoders], dim=1)
        preds_tilda = torch.cat([pred_net(obj_tuples_tilda, temp) for pred_net in self.predicate_encoders], dim=1)

        preds_reshape = torch.zeros(size=(state.size()[0], n_obj.max()**2, P)).to(device)
        preds_next_reshape = torch.zeros(size=(state_next.size()[0], n_obj.max()**2, P)).to(device)
        preds_tilda_reshape = torch.zeros(size=(state_tilda.size()[0], n_obj.max()**2, P)).to(device)

        start_idx = 0
        for i, n in enumerate(n_obj):
            preds_reshape[i, :n**2, :] = preds[start_idx: start_idx+n**2]
            preds_next_reshape[i, :n ** 2, :] = preds_next[start_idx: start_idx+n**2]
            preds_tilda_reshape[i, :n ** 2, :] = preds_tilda[start_idx: start_idx+n**2]
            start_idx += n**2

        return preds_reshape, preds_next_reshape, preds_tilda_reshape


class ActionEncoder(nn.Module):

    def __init__(self):
        super(ActionEncoder, self).__init__()
        self.state_action_encoder = BaseObjectImageEncoder(in_objects=A+ACTION_A, out_features=1)
        self.step_func = TrinaryStep()

    def forward(self, input, temp):
        state, action = input
        obj_action_tuples = self.enumerate_state_action_tuples(state, action)
        logits = self.state_action_encoder(obj_action_tuples)
        logits = logits.view(-1, 1)
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
        self.action_models = nn.ModuleList([ActionEncoder() for _ in range(P)])

    def forward(self, input, temp):

        all_changes = []

        for model in self.action_models:
            preds_change = model(input, temp)
            all_changes.append(preds_change)

        all_changes = torch.stack(all_changes, dim=1)

        return all_changes