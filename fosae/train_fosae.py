from c_swm.utils import StateTransitionsDataset, Concat
from fosae.modules import FoSae, OBJS, STACKS, REMOVE_BG
from fosae.gumble import device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import sys
import os


TEMP_BEGIN = 5
TEMP_MIN = 0.1
ANNEAL_RATE = 0.001
TRAIN_BZ = 183
TEST_BZ = 183

BETA = 1
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

def probs_metric(probs, probs_next, probs_tilda):
    return torch.abs(0.5 - probs).mean().detach(), \
           torch.abs(0.5 - probs_next).mean().detach(), \
           torch.abs(0.5 - probs_tilda).mean().detach()

def preds_similarity_metric(preds, preds_next, preds_tilda, criterion=nn.L1Loss(reduction='none')):
    sum_dim = [i for i in range(1, preds_next.dim())]
    l1_1 = criterion(preds, preds_next).sum(dim=sum_dim).mean()
    l1_2 = criterion(preds, preds_tilda).sum(dim=sum_dim).mean()
    # preds = preds.squeeze()
    # preds_next = preds_next.squeeze()
    # print(preds)
    # print(preds_next)
    # exit(0)
    return l1_1, l1_2

def contrastive_loss_function(pred, preds_tilda, criterion=nn.MSELoss(reduction='none')):
    sum_dim = [i for i in range(1, pred.dim())]
    mse = criterion(pred, preds_tilda).sum(dim=sum_dim).mean()
    return torch.max(torch.tensor(0.0).to(device), torch.tensor(MARGIN).to(device) - mse) * BETA


def epoch_routine(dataloader, vae, temp, optimizer=None):
    if optimizer is not None:
        vae.train()
    else:
        vae.eval()

    contrastive_loss = 0
    metric_pred = 0
    metric_pred_next = 0
    metric_pred_tilda = 0
    pred_sim_metric_1 = 0
    pred_sim_metric_2 = 0

    for i, data in enumerate(dataloader):
        _, _, obj_mask, next_obj_mask, _, _, _, n_obj, _, _, obj_mask_tilda, _ = data
        data = obj_mask.to(device)
        data_next = next_obj_mask.to(device)
        data_tilda = obj_mask_tilda.to(device)
        n_obj = n_obj.to(device)

        # noise1 = torch.normal(mean=0, std=0.2, size=data.size()).to(device)
        # noise2 = torch.normal(mean=0, std=0.2, size=data_next.size()).to(device)
        # noise3 = torch.normal(mean=0, std=0.2, size=data_tilda.size()).to(device)
        noise1 = 0
        noise2 = 0
        noise3 = 0

        if optimizer is None:
            with torch.no_grad():
                preds = vae((data+noise1, data_next+noise2, data_tilda+noise3, n_obj), temp)
                m1, m2, m3 = probs_metric(preds[0], preds[1], preds[2])
                m4, m5 = preds_similarity_metric(preds[0], preds[1], preds[2])
                ctrs_loss = contrastive_loss_function(preds[0], preds[2])
        else:
            preds = vae((data + noise1, data_next + noise2, data_tilda + noise3, n_obj), temp)
            m1, m2, m3 = probs_metric(preds[0], preds[1], preds[2])
            m4, m5 = preds_similarity_metric(preds[0], preds[1], preds[2])
            ctrs_loss = contrastive_loss_function(preds[0], preds[2])

            loss = ctrs_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        contrastive_loss += ctrs_loss.item()
        metric_pred += m1.item()
        metric_pred_next += m2.item()
        metric_pred_tilda += m3.item()
        pred_sim_metric_1 += m4.item()
        pred_sim_metric_2 += m5.item()


    print("{:.2f}, | {:.2f}, {:.2f}, {:.2f}, | {:.2f}, {:.2f}".format
        (
            contrastive_loss / len(dataloader),
            metric_pred / len(dataloader),
            metric_pred_next / len(dataloader),
            metric_pred_tilda / len(dataloader),
            pred_sim_metric_1 / len(dataloader),
            pred_sim_metric_2 / len(dataloader)
        )
    )

    return contrastive_loss / len(dataloader)


def run(n_epoch):
    # train_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks-4-4-det_train.h5", n_obj=9)
    # test_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks-4-4-det_eval.h5", n_obj=9)
    # print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    # train_set = StateTransitionsDataset(hdf5_file="c_swm/data/{}_all.h5".format(PREFIX), n_obj=OBJS+STACKS, remove_bg=REMOVE_BG)
    train_set = Concat(
        [StateTransitionsDataset(
            hdf5_file="c_swm/data/blocks-{}-{}-det_all.h5".format(OBJS, STACKS), n_obj=OBJS + STACKS, remove_bg=True, max_n_obj=8
        ) for OBJS in [1,2,3,4]]
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



