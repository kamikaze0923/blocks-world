from c_swm.utils import StateTransitionsDataset, StateTransitionsDatasetWithLatent, Concat
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
TRAIN_BZ = 108
TEST_BZ = 108
ALPHA = 1

os.makedirs("fosae/model_{}".format(PREFIX), exist_ok=True)
print("Model is FoSae_Action")
ACTION_MODEL_NAME = "FoSae_Action"
print("Training Action Model")


def get_new_dataset(dataloader, vae):

    all_data = []
    all_preds = []
    all_preds_next = []
    all_action_mov_obj_index = []
    all_action_from_obj_index = []
    all_action_tar_obj_index = []
    all_n_obj = []

    for i, data in enumerate(dataloader):
        _, _, obj_mask, next_obj_mask, action_mov_obj_index, action_from_obj_index, action_tar_obj_index, \
        n_obj, _, _, obj_mask_tilda, _ = data

        data = obj_mask.to(device)
        data_next = next_obj_mask.to(device)
        data_tilda = obj_mask_tilda.to(device)

        with torch.no_grad():
            preds, preds_next, _ = vae((data, data_next, data_tilda, n_obj), 0)

        all_data.append(data.cpu())
        all_preds.append(preds.cpu())
        all_preds_next.append(preds_next.cpu())
        all_action_mov_obj_index.append(action_mov_obj_index)
        all_action_from_obj_index.append(action_from_obj_index)
        all_action_tar_obj_index.append(action_tar_obj_index)
        all_n_obj.append(n_obj)

    new_dataset = StateTransitionsDatasetWithLatent(
        (
            torch.cat(all_data, dim=0),
            torch.cat(all_action_mov_obj_index, dim=0),
            torch.cat(all_action_from_obj_index, dim=0),
            torch.cat(all_action_tar_obj_index, dim=0),
            torch.cat(all_preds, dim=0),
            torch.cat(all_preds_next, dim=0),
            torch.cat(all_n_obj, dim=0)
        )
    )

    return new_dataset

# Action similarity in latent space
def action_loss_function(preds_next, preds_next_by_action, criterion=nn.L1Loss(reduction='none')):
    sum_dim = [i for i in range(1, preds_next.dim())]
    l1 = criterion(preds_next_by_action, preds_next).sum(dim=sum_dim).mean()
    return l1 * ALPHA

def preds_similarity_metric(preds, preds_next, criterion=nn.L1Loss(reduction='none')):
    sum_dim = [i for i in range(1, preds_next.dim())]
    l1 = criterion(preds, preds_next).sum(dim=sum_dim).mean()
    return l1

def epoch_routine(dataloader, action_model, temp, optimizer=None):
    if optimizer is not None:
        action_model.train()
    else:
        action_model.eval()

    action_loss = 0
    pred_sim_metric = 0

    for i, data in enumerate(dataloader):
        obj_mask, action_mov_obj_index, action_from_obj_index, action_tar_obj_index, preds, preds_next, n_obj = data
        data = obj_mask.to(device)
        n_obj = n_obj.to(device)

        action_idx = torch.cat([action_mov_obj_index, action_from_obj_index, action_tar_obj_index], dim=1).to(device)
        batch_idx = torch.arange(action_idx.size()[0])
        batch_idx = torch.stack([batch_idx for _ in range(action_idx.size()[1])], dim=1).to(device)
        action = obj_mask[batch_idx, action_idx, : , :, :].to(device)

        noise1 = torch.normal(mean=0, std=0.2, size=data.size()).to(device)
        noise2 = torch.normal(mean=0, std=0.2, size=action.size()).to(device)

        preds = preds.to(device)
        preds_next = preds_next.to(device)

        if optimizer is None:
            with torch.no_grad():
                changes = action_model((data+noise1, action+noise2, n_obj), temp)
                act_loss = action_loss_function(preds_next, preds+changes)
                m1 = preds_similarity_metric(preds, preds_next)
        else:
            changes = action_model((data + noise1, action + noise2, n_obj), temp)
            act_loss = action_loss_function(preds_next, preds+changes)
            m1 = preds_similarity_metric(preds, preds_next)
            loss = act_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        action_loss += act_loss.item()
        pred_sim_metric += m1.item()

    print("{:.2f}, | {:.2f}".format
        (
            action_loss / len(dataloader),
            pred_sim_metric / len(dataloader)
        )
    )

    return action_loss / len(dataloader)


def run(n_epoch):
    train_set = Concat(
        [StateTransitionsDataset(
            hdf5_file="c_swm/data/blocks-{}-{}-det_all.h5".format(OBJS, STACKS), n_obj=OBJS + STACKS, remove_bg=False, max_n_obj=9
        ) for OBJS in [1,2]]
    )
    print("Training Examples: {}".format(len(train_set)))
    assert len(train_set) % TRAIN_BZ == 0
    # assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True)
    vae = FoSae().to(device)
    vae.load_state_dict(torch.load("fosae/model_{}/{}.pth".format(PREFIX, FOSAE_MODEL_NAME), map_location='cpu'))
    print("fosae/model_{}/{}.pth Loaded".format(PREFIX, FOSAE_MODEL_NAME))
    vae.eval()

    new_data_set = get_new_dataset(train_loader, vae)

    del vae
    train_loader = DataLoader(dataset=new_data_set, batch_size=TRAIN_BZ)
    test_loader = DataLoader(dataset=new_data_set, batch_size=TEST_BZ)

    action_model = FoSae_Action().to(device)
    try:
        action_model.load_state_dict(torch.load("fosae/model_{}/{}.pth".format(PREFIX, ACTION_MODEL_NAME), map_location='cpu'))
        print("Action Model Loaded")
    except:
        print("Action Model Loaded Fail")
        pass
    optimizer = Adam(action_model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    # optimizer = SGD(action_model.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1 if e < 100 else 0.1)
    best_loss = float('inf')
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}, Lr: {}".format(e, temp, scheculer.get_last_lr()))
        sys.stdout.flush()
        train_loss = epoch_routine(train_loader, action_model, temp, optimizer)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = epoch_routine(test_loader, action_model, temp)
        print('====> Epoch: {} Average test loss: {:.4f}, Best Test loss: {:.4f}'.format(e, test_loss, best_loss))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(action_model.state_dict(), "fosae/model_{}/{}.pth".format(PREFIX, ACTION_MODEL_NAME))
            best_loss = test_loss
        scheculer.step()


if __name__ == "__main__":
    run(50000)



