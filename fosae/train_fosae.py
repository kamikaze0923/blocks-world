from c_swm.utils import StateTransitionsDataset, Concat
from fosae.modules import FoSae, STACKS, TRAIN_DATASETS_OBJS, MAX_N
from fosae.gumble import device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import sys
import os

TEMP_BEGIN = 1
TEMP_MIN = 0.01
ANNEAL_RATE = 0.005
TRAIN_BZ = 2
TEST_BZ = 108
MARGIN = 1

print("Model is FOSAE")
MODEL_NAME = "FoSae"

# PREFIX = "blocks-{}-{}-det".format(OBJS, STACKS)
PREFIX = "blocks-{}-{}-det".format('all', 'size')
os.makedirs("fosae/model_{}".format(PREFIX), exist_ok=True)
print("Training Encoder and Decoder")

torch.manual_seed(0)

# Reconstruction
def rec_loss_function(recon_x, x, criterion=nn.BCELoss(reduction='none')):
    sum_dim = [i for i in range(1, x.dim())]
    BCE = criterion(recon_x, x).sum(dim=sum_dim).mean()
    return BCE

def probs_metric(probs, probs_next, change):
    return torch.abs(0.5 - probs).mean().detach(), \
           torch.abs(0.5 - probs_next).mean().detach(), \
           torch.abs(change).mean().detach()

def preds_similarity_metric(preds, preds_next, criterion=nn.L1Loss(reduction='none')):
    sum_dim = [i for i in range(1, preds_next.dim())]
    return criterion(preds, preds_next).sum(dim=sum_dim).mean()

def action_supervision_loss(
        pred, pred_next, change, supervision,
        criterion_1=nn.BCELoss(reduction='none'),
        criterion_2=nn.MSELoss(reduction='none'),
        criterion_3=nn.L1Loss(reduction='none')
    ):
    n_pred = set([i.item() for i in torch.arange(pred_next.size()[1])])
    pre_ind, pre_label, eff_ind, eff_label = supervision
    pred_selected = torch.gather(pred, dim=1, index=pre_ind)
    pred_next_selected = torch.gather(pred_next, dim=1, index=eff_ind)
    p1_loss = criterion_1(pred_selected, pre_label).sum(dim=1).mean() + criterion_1(pred_next_selected, eff_label).sum(dim=1).mean()
    all_ind = torch.cat([pre_ind, eff_ind], dim=1)
    p2_loss = 0
    for p, p_n, a_idx in zip(pred, pred_next, all_ind):
        u_ind = a_idx.unique()
        diff = torch.tensor(list(n_pred.difference([i.item() for i in u_ind]))).to(device)
        pred_unchange = torch.index_select(pred, dim=1, index=diff)
        pred_next_unchange = torch.index_select(pred_next, dim=1, index=diff)
        p2_loss += criterion_2(pred_unchange, pred_next_unchange).sum(dim=1).mean()
    a_loss = criterion_3(torch.round(pred_next - pred).detach(), change).sum(dim=1).mean()

    return p1_loss, p2_loss, a_loss

def has_grad(*tensor):
    b = [(t != 0).any() for t in tensor]
    if len(b) == 1:
        print(b[0])
    else:
        assert len(b) == 2
        print(torch.logical_or(b[0], b[1]))


def epoch_routine(dataloader, vae, temp, optimizer=None):
    if optimizer is not None:
        vae.train()
    else:
        vae.eval()

    margin_loss = 0
    transition_loss = 0
    predicate_supervision_loss = 0
    predicate_similarity_loss = 0
    action_loss = 0
    metric_pred = 0
    metric_pred_next = 0
    metric_change = 0
    pred_sim_metric_1 = 0

    for i, data in enumerate(dataloader):
        _, _, obj_mask, next_obj_mask, obj_background, next_obj_background, action_mov_obj_index, action_from_obj_index, action_tar_obj_index,\
        state_n_obj, _, _, _, _, _, _ = data

        data = obj_mask.to(device)
        data_next = next_obj_mask.to(device)
        state_n_obj = state_n_obj.to(device)

        back_grounds = torch.cat([obj_background, next_obj_background], dim=1).to(device)

        # noise1 = torch.normal(mean=0, std=0.2, size=data.size()).to(device)
        # noise2 = torch.normal(mean=0, std=0.2, size=data_next.size()).to(device)
        # noise3 = torch.normal(mean=0, std=0.2, size=data_tilda.size()).to(device)

        noise1 = 0
        noise2 = 0
        noise3 = 0
        noise4 = 0

        action_idx = torch.cat([action_mov_obj_index, action_from_obj_index, action_tar_obj_index], dim=1).to(device)
        action_types = torch.zeros(size=(action_idx.size()[0],), dtype=torch.int32).to(device)
        action_n_obj = torch.tensor([3 for _ in range(action_idx.size()[0])]).to(device)
        action_input = (action_idx, action_n_obj, action_types)
        supervision = vae.get_supervise_signal(action_input)

        if optimizer is None:
            with torch.no_grad():
                preds, change = vae((data+noise1, data_next+noise2, state_n_obj, back_grounds+noise4), action_input, temp)
                preds, preds_next = preds
                m1, m2, m3 = probs_metric(preds, preds_next, change)
                m4 = preds_similarity_metric(preds, preds_next)
                p1_loss, p2_loss, a_loss = action_supervision_loss(preds, preds_next, change, supervision)
        else:
            preds, change = vae((data+noise1, data_next+noise2, state_n_obj, back_grounds+noise4), action_input, temp)
            preds, preds_next = preds
            # print(preds_next.view(TRAIN_BZ, MAX_N + 1, MAX_N))
            # print(preds.view(TRAIN_BZ, MAX_N + 1, MAX_N))
            m1, m2, m3 = probs_metric(preds, preds_next, change)
            m4 = preds_similarity_metric(preds, preds_next)
            # m_loss, t_loss = contrastive_loss_function(preds, preds_next, preds_tilda, change)
            p1_loss, p2_loss, a_loss = action_supervision_loss(preds, preds_next, change, supervision)

            loss = p1_loss + p2_loss + a_loss
            optimizer.zero_grad()
            loss.backward()
            # print("-"*20 + "action semantics grad" + "-"*20)
            # has_grad(vae.action_encoders.action_semantic_encoder[0].fc1.weight.grad)
            # print("-"*20 + "state semantics grad" + "-"*20)
            # has_grad(vae.state_encoders.state_semantic_encoder[0][0].fc1.weight.grad, vae.state_encoders.state_semantic_encoder[1][0].fc1.weight.grad)
            # print("-"*20 + "change predictor grad" + "-"*20)
            # has_grad(vae.state_encoders.state_change_predictor[0][0].fc1.weight.grad, vae.state_encoders.state_change_predictor[1][0].fc1.weight.grad)
            optimizer.step()
            # print(vae.action_encoders.action_semantic_encoder[0].fc1.weight.grad)


        # margin_loss += m_loss.item()
        # transition_loss += t_loss.item()
        predicate_supervision_loss += p1_loss.item()
        predicate_similarity_loss += p2_loss.item()
        action_loss += a_loss.item()

        metric_pred += m1.item()
        metric_pred_next += m2.item()
        metric_change += m3.item()
        pred_sim_metric_1 += m4.item()

    print("{:.2f}, {:.2f}, {:.2f} | {:.2f}, {:.2f}, {:.2f}| {:.2f}".format
        (
            # margin_loss / len(dataloader),
            # transition_loss / len(dataloader),
            predicate_supervision_loss / len(dataloader),
            predicate_similarity_loss / len(dataloader),
            action_loss / len(dataloader),
            metric_pred / len(dataloader),
            metric_pred_next / len(dataloader),
            metric_change / len(dataloader),
            pred_sim_metric_1 / len(dataloader)
        )
    )
    # if optimizer is not None:
    #     print(torch.round(preds_next.view(TRAIN_BZ, MAX_N + 1, MAX_N) - preds.view(TRAIN_BZ, MAX_N + 1, MAX_N)))
    #     print(change.view(TRAIN_BZ, MAX_N + 1, MAX_N))
    #     time.sleep(4)

    return (predicate_supervision_loss + predicate_similarity_loss + action_loss) / len(dataloader)


def run(n_epoch):
    # train_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks-4-4-det_train.h5", n_obj=9)
    # test_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks-4-4-det_eval.h5", n_obj=9)
    # print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    # train_set = StateTransitionsDataset(hdf5_file="c_swm/data/{}_all.h5".format(PREFIX), n_obj=OBJS+STACKS, remove_bg=REMOVE_BG)
    train_set = Concat(
        [StateTransitionsDataset(
            hdf5_file="c_swm/data/blocks-{}-{}-{}_all.h5".format(OBJS, STACKS, 0),
            n_obj=OBJS + STACKS, max_n_obj=MAX_N
        ) for OBJS in TRAIN_DATASETS_OBJS]
    )
    print("Training Examples: {}".format(len(train_set)))
    assert len(train_set) % TRAIN_BZ == 0
    # assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True)
    vae = FoSae().to(device)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    # optimizer = SGD(vae.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1 if e < 100 else 0.1)
    best_loss = float('inf')
    best_tmp = 0
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}, Lr: {}".format(e, temp, scheculer.get_last_lr()))
        sys.stdout.flush()
        train_loss = epoch_routine(train_loader, vae, temp, optimizer)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = epoch_routine(train_loader, vae, temp)
        print('====> Epoch: {} Average test loss: {:.4f}, Best Test loss: {:.4f}, Best temp {:.4f}'.format(e, test_loss, best_loss, best_tmp))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(vae.state_dict(), "fosae/model_{}/{}.pth".format(PREFIX, MODEL_NAME))
            best_loss = test_loss
            best_tmp = temp
        scheculer.step()

if __name__ == "__main__":
    run(50000)



