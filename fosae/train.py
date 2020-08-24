from c_swm.utils import StateTransitionsDataset
from fosae.modules import FoSae
from fosae.gumble import device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

TEMP_BEGIN = 5
TEMP_MIN = 0.7
ANNEAL_RATE = 0.03
TRAIN_BZ = 200
TEST_BZ = 760


print("Model is FOSAE")
MODEL_NAME = "FoSae"


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, criterion=nn.BCELoss(reduction='none')):
    sum_dim = [i for i in range(1, x.dim())]
    # x_min, x_max, rec_x_min, rec_x_max = x.min().item(), x.max().item(), recon_x.min().detach().item(), recon_x.max().detach().item()
    # print("{}-{}-{}-{}".format(x_min, x_max, rec_x_min, rec_x_max))
    # print("{}-{}-{}-{}".format(x_min < 0, x_max > 1, rec_x_min < 0, rec_x_max > 1))
    BCE = criterion(recon_x, x).sum(dim=sum_dim).mean()
    return BCE

def train(dataloader, vae, temp, optimizer):
    vae.train()
    train_loss = 0
    for i, data in enumerate(dataloader):
        if i % 5 == 0:
            print(i*TRAIN_BZ)
        _, _, _, obj_mask, _, _, _ = data
        data = obj_mask.view(obj_mask.size()[0], obj_mask.size()[1], -1)
        data = data.to(device)
        optimizer.zero_grad()
        noise = torch.normal(mean=0, std=0.4, size=data.size()).to(device)
        recon_batch, _ , _ = vae(data+noise, temp)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(dataloader)

def test(dataloader, vae, temp=0):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            _, _, _, obj_mask, _, _, _ = data
            data = obj_mask.view(obj_mask.size()[0], obj_mask.size()[1], -1)
            data = data.to(device)
            recon_batch, _, _ = vae(data, temp)
            loss = loss_function(recon_batch, data)
            test_loss += loss.item()
    return test_loss / len(dataloader)

def load_model(vae):
    vae.load_state_dict(torch.load("fosae/model/{}.pth".format(MODEL_NAME)))
    print("fosae/model/{}.pth loaded".format(MODEL_NAME))



def run(n_epoch):
    train_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks_train.h5", n_obj=9)
    test_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks_eval.h5", n_obj=9)
    print("Training Examples: {}, Testing Examples: {}".format(len(train_set), len(test_set)))
    assert len(train_set) % TRAIN_BZ == 0
    assert len(test_set) % TEST_BZ == 0
    train_loader = DataLoader(train_set, batch_size=TRAIN_BZ, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=TEST_BZ, shuffle=True)
    vae = eval(MODEL_NAME)().to(device)
    load_model(vae)
    optimizer = Adam(vae.parameters(), lr=1e-3)
    scheculer = LambdaLR(optimizer, lambda e: 1.0 if e < 10 else 0.1)
    best_loss = float('inf')
    for e in range(n_epoch):
        temp = np.maximum(TEMP_BEGIN * np.exp(-ANNEAL_RATE * e), TEMP_MIN)
        print("Epoch: {}, Temperature: {}, Lr: {}".format(e, temp, scheculer.get_lr()))
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



