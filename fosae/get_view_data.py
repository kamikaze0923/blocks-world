from c_swm.utils import StateTransitionsDataset
from fosae.modules import FoSae
from fosae.gumble import device
import numpy as np
from torch.utils.data import DataLoader
import torch
from fosae.modules import IMG_H, IMG_W, IMG_C, A, N

N_EXAMPLES = 200
print("Model is FOSAE")
MODEL_NAME = "FoSae"

def init():
    test_set = StateTransitionsDataset(hdf5_file="c_swm/data/blocks_eval.h5", n_obj=9)
    print("View examples {}".format(len(test_set)))

    view_loader = DataLoader(test_set, batch_size=N_EXAMPLES, shuffle=True)
    vae = eval(MODEL_NAME)().to(device)
    vae.load_state_dict(torch.load("fosae/model/{}.pth".format(MODEL_NAME), map_location='cpu'))
    vae.eval()

    return vae, view_loader

def run(vae, view_loader):

    with torch.no_grad():

        data = view_loader.__iter__().__next__()

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

        recon_batch, args, preds = vae((data, data_next, action), 0)

        data_np = data.view(-1, N, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        recon_batch_np = recon_batch[0].view(-1, N, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        args_np = args[0].view(-1, N , A, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        preds_np = preds[0].detach().cpu().numpy()

        print(data_np.shape, recon_batch_np.shape, args_np.shape, preds_np.shape)
        np.save("fosae/block_data/block_data.npy", data_np)
        np.save("fosae/block_data/block_rec.npy", recon_batch_np)
        np.save("fosae/block_data/block_args.npy", args_np)
        np.save("fosae/block_data/block_preds.npy", preds_np)

        data_np = data_next.view(-1, N, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        recon_batch_np = recon_batch[1].view(-1, N, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        args_np = args[1].view(-1, N , A, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        preds_np = preds[1].detach().cpu().numpy()

        print(data_np.shape, recon_batch_np.shape, args_np.shape, preds_np.shape)
        np.save("fosae/block_data/block_data_next.npy", data_np)
        np.save("fosae/block_data/block_rec_next.npy", recon_batch_np)
        np.save("fosae/block_data/block_args_next.npy", args_np)
        np.save("fosae/block_data/block_preds_next.npy", preds_np)

        action_np = preds[2].detach().cpu().numpy()
        print(action_np.shape)
        np.save("fosae/block_data/block_action.npy", action_np)















if __name__ == "__main__":
    vae, view_loader = init()
    run(vae, view_loader)
