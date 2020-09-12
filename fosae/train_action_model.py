from c_swm.utils import StateTransitionsDataset
from fosae.modules import FoSae, FoSae_Action
from fosae.gumble import device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import sys
from fosae.train_fosae import PREFIX, OBJS, STACKS, REMOVE_BG
from fosae.train_fosae import MODEL_NAME as FOSAE_MODEL_NAME
import os

TEMP_BEGIN = 5
TEMP_MIN = 0.01
ANNEAL_RATE = 0.003
TRAIN_BZ = 40
ALPHA = 1

os.makedirs("fosae/model_{}".format(PREFIX), exist_ok=True)
print("Model is FoSae_Action")
ACTION_MODEL_NAME = "FoSae_Action"
print("Training Action Model")


# Action similarity in latent space
def action_loss_function(preds_next, preds_next_by_action, criterion=nn.L1Loss(reduction='none')):
    sum_dim = [i for i in range(1, preds_next.dim())]
    l1 = criterion(preds_next_by_action, preds_next).sum(dim=sum_dim).mean()
    return l1 * ALPHA

def probs_metric(probs, probs_next):
    return torch.abs(0.5 - probs).mean().detach(), torch.abs(0.5 - probs_next).mean().detach()

def epoch_routine(dataloader, vae, action_model, temp, optimizer=None):
    if optimizer is not None:
        action_model.train()
    else:
        action_model.eval()

    action_loss = 0

    for i, data in enumerate(dataloader):
        _, _, _, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index = data
        data = obj_mask.to(device)
        data_next = next_obj_mask.to(device)

        action_idx = torch.cat([action_mov_obj_index, action_tar_obj_index], dim=1).to(device)
        batch_idx = torch.arange(action_idx.size()[0])
        batch_idx = torch.stack([batch_idx, batch_idx], dim=1).to(device)
        action = obj_mask[batch_idx, action_idx, : , :, :].to(device)

        noise1 = torch.normal(mean=0, std=0.2, size=data.size()).to(device)
        noise2 = torch.normal(mean=0, std=0.2, size=action.size()).to(device)

        if optimizer is None:
            with torch.no_grad():
                changes = action_model((data+noise1, action+noise2), temp)

                _, preds_all = vae((data, data_next), 0)
                preds, preds_next = preds_all

                act_loss = action_loss_function(preds_next, preds+changes)
        else:
            changes = action_model((data + noise1, action + noise2), temp)

            with torch.no_grad():
                _, preds_all = vae((data, data_next), 0)
            preds, preds_next = preds_all

            act_loss = action_loss_function(preds_next, preds + changes)
            loss = act_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        action_loss += act_loss.item()

    return action_loss / len(dataloader)

def run(n_epoch):
    train_set = StateTransitionsDataset(hdf5_file="c_swm/data/{}_all.h5".format(PREFIX), n_obj=OBJS+STACKS, remove_bg=REMOVE_BG)
    print("Training Examples: {}".format(len(train_set)))
    assert len(train_set) % TRAIN_BZ == 0
    # assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True)
    vae = FoSae().to(device)
    vae.load_state_dict(torch.load("fosae/model_{}/{}.pth".format(PREFIX, FOSAE_MODEL_NAME), map_location='cpu'))
    vae.eval()

    action_model = FoSae_Action().to(device)
    optimizer = Adam(action_model.parameters(), lr=1e-3)
    # optimizer = SGD(action_model.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1 if e < 100 else 0.1)
    best_loss = float('inf')
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}, Lr: {}".format(e, temp, scheculer.get_last_lr()))
        sys.stdout.flush()
        train_loss = epoch_routine(train_loader, vae, action_model, temp, optimizer)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = epoch_routine(train_loader, vae, action_model, temp)
        print('====> Epoch: {} Average test loss: {:.4f}, Best Test loss: {:.4f}'.format(e, test_loss, best_loss))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(action_model.state_dict(), "fosae/model_{}/{}.pth".format(PREFIX, ACTION_MODEL_NAME))
            best_loss = test_loss
        scheculer.step()


if __name__ == "__main__":
    run(50000)



