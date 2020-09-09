from c_swm.utils import StateTransitionsDataset
from fosae.modules import FoSae, OBJS, STACKS, REMOVE_BG
from fosae.gumble import device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import sys
import pickle


TEMP_BEGIN = 20
TEMP_MIN = 0.7
ANNEAL_RATE = 0.0001
TRAIN_BZ = 12
TEST_BZ = 12
ALPHA = 1
BETA = 1
MARGIN = 1

print("Model is FOSAE")
MODEL_NAME = "FoSae"

PREFIX = "blocks-{}-{}-det".format(OBJS, STACKS)

TRAIN_ACTION_MODEL = False
if TRAIN_ACTION_MODEL:
    TEMP_BEGIN = pickle.load(open("fosae/model/metafile.pkl", 'rb'))['temp']
    print("Training Action Model, temp begin {}".format(TEMP_BEGIN))
else:
    print("Training Encoder and Decoder")

# Reconstruction
def rec_loss_function(recon_x, x, criterion=nn.BCELoss(reduction='none')):
    sum_dim = [i for i in range(1, x.dim())]
    BCE = criterion(recon_x, x).sum(dim=sum_dim).mean()
    return BCE

# Action similarity in latent space
def action_loss_function(preds_next, preds_next_by_action, criterion=nn.MSELoss(reduction='none')):
    sum_dim = [i for i in range(1, preds_next.dim())]
    mse = criterion(preds_next_by_action, preds_next.detach()).sum(dim=sum_dim).mean()
    return mse * ALPHA, torch.abs(0.5 - preds_next).sum(dim=-1).mean().detach(), torch.abs(0.5 - preds_next_by_action).sum(dim=-1).mean().detach()

def probs_metric(probs, probs_next):
    return torch.abs(0.5 - probs).mean().detach(), torch.abs(0.5 - probs_next).mean().detach()


def contrastive_loss_function(pred, preds_next, criterion=nn.MSELoss(reduction='none')):
    sum_dim = [i for i in range(1, pred.dim())]
    mse = criterion(pred, preds_next).sum(dim=sum_dim).mean()
    return torch.max(torch.tensor(0.0).to(device), torch.tensor(MARGIN).to(device) - mse) * BETA

def epoch_routine(dataloader, vae, temp, optimizer=None):
    if optimizer is not None:
        vae.train()
    else:
        vae.eval()
    recon_loss0 = 0
    recon_loss1 = 0
    recon_loss2 = 0
    action_loss = 0
    contrastive_loss = 0
    metric_pred = 0
    metric_pred_next = 0
    metric_prob = 0
    metric_prob_next = 0


    for i, data in enumerate(dataloader):
        _, _, _, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index = data
        data = obj_mask.to(device)
        data_next = next_obj_mask.to(device)

        action_idx = torch.cat([action_mov_obj_index, action_tar_obj_index], dim=1).to(device)
        batch_idx = torch.arange(action_idx.size()[0])
        batch_idx = torch.stack([batch_idx, batch_idx], dim=1).to(device)
        action = obj_mask[batch_idx, action_idx, : , :, :].to(device)

        noise1 = torch.normal(mean=0, std=0.2, size=data.size()).to(device)
        noise2 = torch.normal(mean=0, std=0.2, size=data_next.size()).to(device)
        noise3 = torch.normal(mean=0, std=0.2, size=action.size()).to(device)

        if optimizer is None:
            with torch.no_grad():
                recon_batch, _ , preds, probs = vae((data+noise1, data_next+noise2, action+noise3), temp)
                rec_loss0 = rec_loss_function(recon_batch[0], data)
                rec_loss1 = rec_loss_function(recon_batch[1], data_next)
                rec_loss2 = rec_loss_function(recon_batch[2], data_next)
                act_loss, m0, m1 = action_loss_function(preds[1], preds[2])
                m3, m4 = probs_metric(probs[0], probs[1])
                ctrs_loss = contrastive_loss_function(preds[0], preds[1])
        else:
            recon_batch, _, preds, probs = vae((data + noise1, data_next + noise2, action + noise3), temp)
            rec_loss0 = rec_loss_function(recon_batch[0], data)
            rec_loss1 = rec_loss_function(recon_batch[1], data_next)
            rec_loss2 = rec_loss_function(recon_batch[2], data_next)
            act_loss, m0, m1 = action_loss_function(preds[1], preds[2])
            m3, m4 = probs_metric(probs[0], probs[1])
            ctrs_loss = contrastive_loss_function(preds[0], preds[1])

            if not TRAIN_ACTION_MODEL:
                loss = rec_loss0 + rec_loss1
            else:
                loss = act_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        recon_loss0 += rec_loss0.item()
        recon_loss1 += rec_loss1.item()
        recon_loss2 += rec_loss2.item()
        action_loss += act_loss.item()
        contrastive_loss += ctrs_loss.item()
        metric_pred += m0.item()
        metric_pred_next += m1.item()
        metric_prob += m3.item()
        metric_prob_next += m4.item()


    print("{:.2f}, {:.2f}, {:.2f}, | {:.2f}, {:.2f}, | {:.2f}, {:.2f}, {:.2f}, {:.2f}".format
        (
            recon_loss0 / len(dataloader),
            recon_loss1 / len(dataloader),
            recon_loss2 / len(dataloader),
            action_loss / len(dataloader),
            contrastive_loss / len(dataloader),
            metric_pred / len(dataloader),
            metric_pred_next / len(dataloader),
            metric_prob / len(dataloader),
            metric_prob_next / len(dataloader)
        )
    )

    if TRAIN_ACTION_MODEL:
        metric = (action_loss) / len(dataloader)
    else:
        metric = (recon_loss0 + recon_loss1) / len(dataloader)

    return metric

def load_model(vae):
    vae.load_state_dict(torch.load("fosae/model/{}.pth".format(MODEL_NAME), map_location='cpu'))
    print("fosae/model/{}.pth loaded".format(MODEL_NAME))

def run(n_epoch, test_only=False):
    meta_file = 'fosae/model/metadata.pkl'
    # train_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks-4-4-det_train.h5", n_obj=9)
    # test_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks-4-4-det_eval.h5", n_obj=9)
    # print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    train_set = StateTransitionsDataset(hdf5_file="c_swm/data/{}_all.h5".format(PREFIX), n_obj=OBJS+STACKS, remove_bg=REMOVE_BG)
    print("Training Examples: {}".format(len(train_set)))
    assert len(train_set) % TRAIN_BZ == 0
    # assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True)
    vae = eval(MODEL_NAME)().to(device)
    if TRAIN_ACTION_MODEL or test_only:
        load_model(vae)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1 if e < 100 else 0.1)
    best_loss = float('inf')
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}, Lr: {}".format(e, temp, scheculer.get_last_lr()))
        sys.stdout.flush()
        if test_only:
            test_loss = epoch_routine(train_loader, vae, temp)
            print('====> Epoch: {} Average test loss: {:.4f}'.format(e, test_loss))
            exit(0)
        train_loss = epoch_routine(train_loader, vae, temp, optimizer)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = epoch_routine(train_loader, vae, temp)
        print('====> Epoch: {} Average test loss: {:.4f}'.format(e, test_loss))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(vae.state_dict(), "fosae/model/{}.pth".format(MODEL_NAME))
            pickle.dump({'temp': temp}, open("fosae/model/metafile.pkl", "wb"))
            best_loss = test_loss
        scheculer.step()



if __name__ == "__main__":
    run(2000, test_only=False)



