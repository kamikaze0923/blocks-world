from c_swm.utils import StateTransitionsDataset, Concat
from fosae.modules import FoSae, STACKS, REMOVE_BG, TRAIN_DATASETS_OBJS, MAX_N
from fosae.gumble import device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import sys
import os

TEMP_BEGIN = 5
TEMP_MIN = 0.1
ANNEAL_RATE = 0.001
TRAIN_BZ = 27
TEST_BZ = 27
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

def probs_metric(probs, probs_next, probs_tilda, change):
    return torch.abs(0.5 - probs).mean().detach(), \
           torch.abs(0.5 - probs_next).mean().detach(), \
           torch.abs(0.5 - probs_tilda).mean().detach(), \
           torch.abs(change).mean().detach()

def preds_similarity_metric(preds, preds_next, preds_tilda, criterion=nn.L1Loss(reduction='none')):
    sum_dim = [i for i in range(1, preds_next.dim())]
    l1_1 = criterion(preds, preds_next).sum(dim=sum_dim).mean()
    l1_2 = criterion(preds, preds_tilda).sum(dim=sum_dim).mean()
    return l1_1, l1_2

def contrastive_loss_function(pred, pred_next, preds_tilda, change, criterion=nn.L1Loss(reduction='none')):
    sum_dim = [i for i in range(1, pred.dim())]
    margin_loss = torch.max(
        torch.tensor(0.0).to(device),
        torch.tensor(MARGIN).to(device) - criterion(pred, preds_tilda).sum(dim=sum_dim).mean()
    )
    transition_loss = criterion(pred_next+change, pred_next).sum(dim=sum_dim).mean()
    return margin_loss, transition_loss

def epoch_routine(dataloader, vae, temp, optimizer=None):
    if optimizer is not None:
        vae.train()
    else:
        vae.eval()

    margin_loss = 0
    transition_loss = 0
    metric_pred = 0
    metric_pred_next = 0
    metric_pred_tilda = 0
    metric_change = 0
    pred_sim_metric_1 = 0
    pred_sim_metric_2 = 0

    for i, data in enumerate(dataloader):
        _, _, obj_mask, next_obj_mask, action_mov_obj_index, action_from_obj_index, action_tar_obj_index,\
        state_n_obj, _, _, obj_mask_tilda, _ = data

        data = obj_mask.to(device)
        data_next = next_obj_mask.to(device)
        data_tilda = obj_mask_tilda.to(device)
        state_n_obj = state_n_obj.to(device)

        # noise1 = torch.normal(mean=0, std=0.2, size=data.size()).to(device)
        # noise2 = torch.normal(mean=0, std=0.2, size=data_next.size()).to(device)
        # noise3 = torch.normal(mean=0, std=0.2, size=data_tilda.size()).to(device)

        noise1 = 0
        noise2 = 0
        noise3 = 0

        action_idx = torch.cat([action_mov_obj_index, action_from_obj_index, action_tar_obj_index], dim=1).to(device)
        action_types = torch.zeros(size=(action_idx.size()[0],), dtype=torch.int32).to(device)
        action_n_obj = torch.tensor([3 for _ in range(action_idx.size()[0])]).to(device)

        if optimizer is None:
            with torch.no_grad():
                preds, change = vae((data+noise1, data_next+noise2, data_tilda+noise3, state_n_obj), (action_idx, action_n_obj, action_types), temp)
                preds, preds_next, preds_tilda = preds
                m1, m2, m3, m4 = probs_metric(preds, preds_next, preds_tilda)
                m5, m6 = preds_similarity_metric(preds, preds_next, preds_tilda)
                m_loss, t_loss = contrastive_loss_function(preds, preds_next, preds_tilda, change)
        else:
            preds, change = vae((data+noise1, data_next+noise2, data_tilda+noise3, state_n_obj), (action_idx, action_n_obj, action_types), temp)
            preds, preds_next, preds_tilda = preds
            m1, m2, m3, m4 = probs_metric(preds, preds_next, preds_tilda)
            m5, m6 = preds_similarity_metric(preds, preds_next, preds_tilda)
            m_loss, t_loss = contrastive_loss_function(preds, preds_next, preds_tilda, change)
            loss = m_loss + t_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        margin_loss += m_loss.item()
        transition_loss += t_loss.item()
        metric_pred += m1.item()
        metric_pred_next += m2.item()
        metric_pred_tilda += m3.item()
        metric_change += m4.item()
        pred_sim_metric_1 += m5.item()
        pred_sim_metric_2 += m6.item()


    print("{:.2f}, {:.2f} | {:.2f}, {:.2f}, {:.2f}, {:.2f} | {:.2f}, {:.2f}".format
        (
            margin_loss / len(dataloader),
            transition_loss / len(dataloader),
            metric_pred / len(dataloader),
            metric_pred_next / len(dataloader),
            metric_pred_tilda / len(dataloader),
            metric_change / len(dataloader),
            pred_sim_metric_1 / len(dataloader),
            pred_sim_metric_2 / len(dataloader)
        )
    )

    return (margin_loss + transition_loss) / len(dataloader)


def run(n_epoch):
    # train_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks-4-4-det_train.h5", n_obj=9)
    # test_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks-4-4-det_eval.h5", n_obj=9)
    # print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    # train_set = StateTransitionsDataset(hdf5_file="c_swm/data/{}_all.h5".format(PREFIX), n_obj=OBJS+STACKS, remove_bg=REMOVE_BG)
    train_set = Concat(
        [StateTransitionsDataset(
            hdf5_file="c_swm/data/blocks-{}-{}-{}_all.h5".format(OBJS, STACKS, 0),
            n_obj=OBJS + STACKS, remove_bg=REMOVE_BG, max_n_obj=MAX_N
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
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}, Lr: {}".format(e, temp, scheculer.get_last_lr()))
        sys.stdout.flush()
        train_loss = epoch_routine(train_loader, vae, temp, optimizer)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = epoch_routine(train_loader, vae, temp)
        print('====> Epoch: {} Average test loss: {:.4f}, Best Test loss: {:.4f}'.format(e, test_loss, best_loss))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(vae.state_dict(), "fosae/model_{}/{}.pth".format(PREFIX, MODEL_NAME))
            best_loss = test_loss
        scheculer.step()

if __name__ == "__main__":
    run(10000)



