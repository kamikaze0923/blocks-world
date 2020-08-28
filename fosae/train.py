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

TEMP_BEGIN = 10
TEMP_MIN = 0.3
ANNEAL_RATE = 0.03
TRAIN_BZ = 180
TEST_BZ = 720
ALPHA = 10
BETA = 1
MARGIN = 1

print("Model is FOSAE")
MODEL_NAME = "FoSae"

# Reconstruction
def rec_loss_function(recon_x, x, criterion=nn.BCELoss(reduction='none')):
    sum_dim = [i for i in range(1, x.dim())]
    BCE = criterion(recon_x, x).sum(dim=sum_dim).mean()
    return BCE

# Action similarity in latent space
def action_loss_function(pred, preds_next, action, criterion=nn.BCELoss(reduction='none'), detach_encoder=True):
    if detach_encoder:
        pred = pred.detach()
        preds_next = preds_next.detach()

    def range_normalize(x):
        return (x + 1.0) / 3.0

    sum_dim = [i for i in range(1, pred.dim())]
    MSE = criterion(range_normalize(pred+action), range_normalize(preds_next)).sum(dim=sum_dim).mean()
    return MSE * ALPHA

def contrastive_loss_function(pred, preds_next, criterion=nn.MSELoss(reduction='none')):
    sum_dim = [i for i in range(1, pred.dim())]
    MSE = criterion(pred, preds_next).sum(dim=sum_dim).mean()
    return torch.max(torch.tensor(0.0).to(device), torch.tensor(MARGIN).to(device) - MSE) * BETA

def train(dataloader, vae, temp, optimizer):
    vae.train()
    recon_loss0 = 0
    recon_loss1 = 0
    action_loss = 0
    contrastive_loss = 0
    for i, data in enumerate(dataloader):
        _, _, _, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index = data
        data = obj_mask.to(device)
        data_next = next_obj_mask.to(device)

        action_idx = torch.cat([action_mov_obj_index, action_tar_obj_index], dim=1).to(device)
        batch_idx = torch.arange(action_idx.size()[0])
        batch_idx = torch.stack([batch_idx, batch_idx], dim=1).to(device)
        action = obj_mask[batch_idx, action_idx, : , :].to(device)

        noise1 = torch.normal(mean=0, std=0.4, size=data.size()).to(device)
        noise2 = torch.normal(mean=0, std=0.4, size=data_next.size()).to(device)
        noise3 = torch.normal(mean=0, std=0.4, size=action.size()).to(device)
        recon_batch, _ , preds = vae((data+noise1, data_next+noise2, action+noise3), temp)

        rec_loss0 = rec_loss_function(recon_batch[0], data)
        rec_loss1 = rec_loss_function(recon_batch[1], data_next)
        act_loss = action_loss_function(*preds)
        ctrs_loss = contrastive_loss_function(preds[0], preds[1])
        loss = rec_loss0 + rec_loss1 + act_loss + ctrs_loss
        optimizer.zero_grad()
        loss.backward()

        recon_loss0 += rec_loss0.item()
        recon_loss1 += rec_loss1.item()
        action_loss += act_loss.item()
        contrastive_loss += ctrs_loss.item()
        optimizer.step()

    print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format
        (
            recon_loss0 / len(dataloader),
            recon_loss1 / len(dataloader),
            action_loss / len(dataloader),
            contrastive_loss / len(dataloader)
        )
    )

    return (recon_loss0 + recon_loss1 + action_loss + contrastive_loss) / len(dataloader)

def test(dataloader, vae, temp=0):
    vae.eval()
    recon_loss0 = 0
    recon_loss1 = 0
    action_loss = 0
    contrastive_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            _, _, _, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index = data
            data = obj_mask.to(device)
            data_next = next_obj_mask.to(device)

            action_idx = torch.cat([action_mov_obj_index, action_tar_obj_index], dim=1).to(device)
            batch_idx = torch.arange(action_idx.size()[0])
            batch_idx = torch.stack([batch_idx, batch_idx], dim=1).to(device)
            action = obj_mask[batch_idx, action_idx, : , :].to(device)

            recon_batch, _, preds = vae((data, data_next, action), temp)

            rec_loss0 = rec_loss_function(recon_batch[0], data)
            rec_loss1 = rec_loss_function(recon_batch[1], data_next)
            act_loss = action_loss_function(*preds)
            ctrs_loss = contrastive_loss_function(preds[0], preds[1])

            recon_loss0 += rec_loss0.item()
            recon_loss1 += rec_loss1.item()
            action_loss += act_loss.item()
            contrastive_loss += ctrs_loss.item()


    print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format
        (
            recon_loss0 / len(dataloader),
            recon_loss1 / len(dataloader),
            action_loss / len(dataloader),
            contrastive_loss / len(dataloader)
        )
    )

    return (recon_loss0 + recon_loss1 + action_loss + contrastive_loss) / len(dataloader)

def load_model(vae):
    vae.load_state_dict(torch.load("fosae/model/{}.pth".format(MODEL_NAME), map_location='cpu'))
    print("fosae/model/{}.pth loaded".format(MODEL_NAME))

def run(n_epoch):
    sys.stdout.flush()
    # train_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks_train.h5", n_obj=9)
    # test_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks_eval.h5", n_obj=9)
    # print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    train_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks_all.h5", n_obj=9)
    print("Training Examples: {}".format(len(train_set)))
    sys.stdout.flush()
    assert len(train_set) % TRAIN_BZ == 0
    # assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    # # test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True)
    vae = eval(MODEL_NAME)().to(device)
    # load_model(vae)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 0.1 if e < 100 else 0.1)
    best_loss = float('inf')
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}, Lr: {}".format(e, temp, scheculer.get_lr()))
        sys.stdout.flush()
        train_loss = train(train_loader, vae, temp, optimizer)
        print('====> Epoch: {} Average train loss: {:.4f}'.format(e, train_loss))
        test_loss = test(train_loader, vae)
        print('====> Epoch: {} Average test loss: {:.4f}'.format(e, test_loss))
        if test_loss < best_loss:
            print("Save Model")
            torch.save(vae.state_dict(), "fosae/model/{}.pth".format(MODEL_NAME))
            best_loss = test_loss
        scheculer.step()



if __name__ == "__main__":
    run(1000)



