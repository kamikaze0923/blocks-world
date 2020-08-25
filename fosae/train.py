from c_swm.utils import StateTransitionsDataset
from fosae.modules import FoSae
from fosae.gumble import device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import sys

TEMP_BEGIN = 5
TEMP_MIN = 0.7
ANNEAL_RATE = 0.03
TRAIN_BZ = 200
TEST_BZ = 190
GAMMA = 100


print("Model is FOSAE")
MODEL_NAME = "FoSae"


# Reconstruction
def rec_loss_function(recon_x, x, criterion=nn.BCELoss(reduction='none')):
    sum_dim = [i for i in range(1, x.dim())]
    BCE = criterion(recon_x, x).sum(dim=sum_dim).mean()
    return BCE

# Action similarity in latent space
def action_loss_function(x, x_next, action, criterion=nn.MSELoss(reduction='none')):
    sum_dim = [i for i in range(1, x.dim())]
    MSE = criterion(x+action, x_next).sum(dim=sum_dim).mean()
    return MSE * GAMMA

def train(dataloader, vae, temp, optimizer):
    vae.train()
    recon_loss = 0
    action_loss = 0
    for i, data in enumerate(dataloader):
        _, _, _, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index = data
        data = obj_mask.view(obj_mask.size()[0], obj_mask.size()[1], -1)
        data = data.to(device)
        data_next = next_obj_mask.view(next_obj_mask.size()[0], next_obj_mask.size()[1], -1)
        data_next = data_next.to(device)

        action_idx = torch.cat([action_mov_obj_index, action_tar_obj_index], dim=1).to(device)
        batch_idx = torch.arange(action_idx.size()[0])
        batch_idx = torch.stack([batch_idx, batch_idx], dim=1).to(device)
        action = obj_mask[batch_idx, action_idx, : , :]
        action = action.view(action.size()[0], action.size()[1], -1).to(device)

        noise1 = torch.normal(mean=0, std=0.4, size=data.size()).to(device)
        noise2 = torch.normal(mean=0, std=0.4, size=data_next.size()).to(device)
        noise3 = torch.normal(mean=0, std=0.4, size=action.size()).to(device)
        recon_batch, _ , preds = vae((data+noise1, data_next+noise2, action+noise3), temp)

        rec_loss0 = rec_loss_function(recon_batch[0], data)
        rec_loss1 = rec_loss_function(recon_batch[1], data_next)
        act_loss = action_loss_function(*preds)
        loss = rec_loss0 + rec_loss1 + act_loss
        optimizer.zero_grad()
        loss.backward()

        recon_loss += rec_loss0.item()
        recon_loss += rec_loss1.item()
        action_loss =+ act_loss.item()
        optimizer.step()
        print("{:.2f}, {:.2f}, {:.2f}".format(rec_loss0.item(), rec_loss1.item(), act_loss.item()))
    return (recon_loss + action_loss) / len(dataloader)

def test(dataloader, vae, temp=0):
    vae.eval()
    recon_loss = 0
    action_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            _, _, _, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index = data
            data = obj_mask.view(obj_mask.size()[0], obj_mask.size()[1], -1)
            data = data.to(device)
            data_next = next_obj_mask.view(next_obj_mask.size()[0], next_obj_mask.size()[1], -1)
            data_next = data_next.to(device)

            action_idx = torch.cat([action_mov_obj_index, action_tar_obj_index], dim=1).to(device)
            batch_idx = torch.arange(action_idx.size()[0])
            batch_idx = torch.stack([batch_idx, batch_idx], dim=1).to(device)
            action = obj_mask[batch_idx, action_idx, :, :]
            action = action.view(action.size()[0], action.size()[1], -1).to(device)


            recon_batch, _, preds = vae((data, data_next, action), temp)

            rec_loss0 = rec_loss_function(recon_batch[0], data)
            rec_loss1 = rec_loss_function(recon_batch[1], data_next)
            act_loss = action_loss_function(*preds)

            recon_loss += rec_loss0.item()
            recon_loss += rec_loss1.item()
            action_loss = + act_loss.item()

            print("{:.2f}, {:.2f}, {:.2f}".format(rec_loss0.item(), rec_loss1.item(), act_loss.item()))

    return (recon_loss+action_loss) / len(dataloader)

def load_model(vae):
    vae.load_state_dict(torch.load("fosae/model/{}.pth".format(MODEL_NAME), map_location='cpu'))
    print("fosae/model/{}.pth loaded".format(MODEL_NAME))



def run(n_epoch):
    sys.stdout.flush()
    train_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks_train.h5", n_obj=9)
    test_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks_eval.h5", n_obj=9)
    print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    sys.stdout.flush()
    assert len(train_set) % TRAIN_BZ == 0
    assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True)
    vae = eval(MODEL_NAME)()
    load_model(vae)
    vae.to(device)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1.0 if e < 10 else 0.1)
    best_loss = float('inf')
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}, Lr: {}".format(e, temp, scheculer.get_lr()))
        sys.stdout.flush()
        train_loss = train(train_loader, vae, temp, optimizer)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = test(test_loader, vae)
        print('====> Epoch: {} Average test loss: {:.4f}'.format(e, test_loss))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(vae.state_dict(), "fosae/model/{}.pth".format(MODEL_NAME))
            best_loss = test_loss
        scheculer.step()



if __name__ == "__main__":
    run(1000)



