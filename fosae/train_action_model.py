from c_swm.utils import StateTransitionsDataset
from fosae.modules import FoSae, FoSae_Action
from fosae.gumble import device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import sys
import pickle
from fosae.train_fosae import PREFIX, OBJS, STACKS, REMOVE_BG
from fosae.train_fosae import MODEL_NAME as FOSAE_MODEL_NAME

TEMP_BEGIN = 5
TEMP_MIN = 0.01
ANNEAL_RATE = 0.003
TRAIN_BZ = 2
ALPHA = 1

print("Model is FoSae_Action")
ACTION_MODEL_NAME = "FoSae_Action"
print("Training Action Model")


# Action similarity in latent space
def action_loss_function(preds_next, preds_next_by_action, criterion=nn.MSELoss(reduction='none')):
    sum_dim = [i for i in range(1, preds_next.dim())]
    mse = criterion(preds_next_by_action, preds_next.detach()).sum(dim=sum_dim).mean()
    return mse * ALPHA

def probs_metric(probs, probs_next):
    return torch.abs(0.5 - probs).mean().detach(), torch.abs(0.5 - probs_next).mean().detach()


def epoch_routine(dataloader, vae, action_model, temp, optimizer=None):
    if optimizer is not None:
        action_model.train()
    else:
        action_model.eval()
    recon_loss0 = 0
    recon_loss1 = 0
    contrastive_loss = 0
    metric_pred = 0
    metric_pred_next = 0

    for i, data in enumerate(dataloader):
        _, _, _, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index = data
        data = obj_mask.to(device)
        data_next = next_obj_mask.to(device)

        action_idx = torch.cat([action_mov_obj_index, action_tar_obj_index], dim=1).to(device)
        batch_idx = torch.arange(action_idx.size()[0])
        batch_idx = torch.stack([batch_idx, batch_idx], dim=1).to(device)
        action = obj_mask[batch_idx, action_idx, : , :, :].to(device)

        with torch.no_grad():
            _, preds_all = vae((data, data_next), 0)
        preds, preds_next = preds_all

        noise1 = torch.normal(mean=0, std=0.2, size=data.size()).to(device)
        noise2 = torch.normal(mean=0, std=0.2, size=data_next.size()).to(device)

        if optimizer is None:
            with torch.no_grad():
                changes = action_model((data+noise1, action+noise2), temp)
                act_loss = action_loss_function(preds_next, preds+changes)
        else:
            recon_batch, preds = vae((data + noise1, data_next + noise2, action + noise3), temp)
            rec_loss0 = rec_loss_function(recon_batch[0], data)
            rec_loss1 = rec_loss_function(recon_batch[1], data_next)
            m1, m2 = probs_metric(preds[0], preds[1])
            ctrs_loss = contrastive_loss_function(preds[0], preds[1])
            loss = rec_loss0 + rec_loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        recon_loss0 += rec_loss0.item()
        recon_loss1 += rec_loss1.item()
        contrastive_loss += ctrs_loss.item()
        metric_pred += m1.item()
        metric_pred_next += m2.item()


    print("{:.2f}, {:.2f} | {:.2f}, | {:.2f}, {:.2f}".format
        (
            recon_loss0 / len(dataloader),
            recon_loss1 / len(dataloader),
            contrastive_loss / len(dataloader),
            metric_pred / len(dataloader),
            metric_pred_next / len(dataloader),
        )
    )

    return (recon_loss0 + recon_loss1) / len(dataloader)


def run(n_epoch):
    train_set = StateTransitionsDataset(hdf5_file="c_swm/data/{}_all.h5".format(PREFIX), n_obj=OBJS+STACKS, remove_bg=REMOVE_BG)
    print("Training Examples: {}".format(len(train_set)))
    assert len(train_set) % TRAIN_BZ == 0
    # assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True)
    vae = FoSae().to(device)
    vae.load_state_dict(torch.load("fosae/model/{}.pth".format(FOSAE_MODEL_NAME), map_location='cpu'))
    vae.eval()

    action_model = FoSae_Action()
    optimizer = Adam(action_model.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1 if e < 100 else 0.1)
    best_loss = float('inf')
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}, Lr: {}".format(e, temp, scheculer.get_last_lr()))
        sys.stdout.flush()
        train_loss = epoch_routine(train_loader, vae, action_model, temp, optimizer)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = epoch_routine(train_loader, vae, temp)
        print('====> Epoch: {} Average test loss: {:.4f}, Best Test loss: {:.4f}'.format(e, test_loss, best_loss))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(vae.state_dict(), "fosae/model/{}.pth".format(ACTION_MODEL_NAME))
            pickle.dump({'temp': temp}, open("fosae/model/metafile_action.pkl", "wb"))
            best_loss = test_loss
        scheculer.step()



if __name__ == "__main__":
    run(20000)



