from c_swm.utils import StateTransitionsDataset
from fosae.modules import FoSae, FoSae_Action
from fosae.gumble import device
import numpy as np
from torch.utils.data import DataLoader
import torch
from fosae.modules import IMG_H, IMG_W, IMG_C, A, N, U, P
from fosae.modules import OBJS, STACKS, REMOVE_BG
from fosae.train_fosae import PREFIX
from fosae.train_fosae import MODEL_NAME as FOSAE_MODEL_NAME
from fosae.train_action_model import ACTION_MODEL_NAME

N_OBJ = OBJS + STACKS + (0 if REMOVE_BG else 1)
N_EXAMPLES = 96




def init():
    test_set = StateTransitionsDataset(hdf5_file="c_swm/data/{}_all.h5".format(PREFIX), n_obj=OBJS+STACKS, remove_bg=REMOVE_BG)
    print("View examples {}".format(len(test_set)))

    view_loader = DataLoader(test_set, batch_size=N_EXAMPLES, shuffle=True)

    vae = FoSae().to(device)
    vae.load_state_dict(torch.load("fosae/model_{}/{}.pth".format(PREFIX, FOSAE_MODEL_NAME), map_location='cpu'))
    vae.eval()

    action_model = FoSae_Action().to(device)
    action_model.load_state_dict(torch.load("fosae/model_{}/{}.pth".format(PREFIX, ACTION_MODEL_NAME), map_location='cpu'))
    action_model.eval()

    return vae, action_model, view_loader

def run(vae, action_model, view_loader):

    with torch.no_grad():

        data = view_loader.__iter__().__next__()

        _, _, _, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index = data
        data = obj_mask.to(device)
        data_next = next_obj_mask.to(device)

        recon_batch, preds = vae((data, data_next), 0)

        data_np = data.view(-1, N, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        recon_batch_np = recon_batch[0].view(-1, N, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        preds_np = preds[0].detach().cpu().numpy().reshape(-1, P, N, N)

        print(data_np.shape, recon_batch_np.shape, preds_np.shape)
        np.save("fosae/block_data/block_data.npy", data_np)
        np.save("fosae/block_data/block_rec.npy", recon_batch_np)
        np.save("fosae/block_data/block_preds.npy", preds_np)

        data_np = data_next.view(-1, N, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        recon_batch_np = recon_batch[1].view(-1, N, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        preds_np = preds[1].detach().cpu().numpy().reshape(-1, P, N, N)

        print(data_np.shape, recon_batch_np.shape, preds_np.shape)
        np.save("fosae/block_data/block_data_next.npy", data_np)
        np.save("fosae/block_data/block_rec_next.npy", recon_batch_np)
        np.save("fosae/block_data/block_preds_next.npy", preds_np)


        action_idx = torch.cat([action_mov_obj_index, action_tar_obj_index], dim=1).to(device)
        batch_idx = torch.arange(action_idx.size()[0])
        batch_idx = torch.stack([batch_idx, batch_idx], dim=1).to(device)
        action = obj_mask[batch_idx, action_idx, : , :, :].to(device)

        actions = action_model((data, action), 0)
        action_np = actions.detach().cpu().numpy().reshape(-1, P, N, N)
        print(action_np.shape)
        np.save("fosae/block_data/action.npy", action_np)




if __name__ == "__main__":
    vae, action_model, view_loader = init()
    run(vae, action_model, view_loader)
