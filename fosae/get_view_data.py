from c_swm.utils import StateTransitionsDataset, Concat
from fosae.modules import FoSae
from fosae.gumble import device
import numpy as np
from torch.utils.data import DataLoader
import torch
from fosae.modules import IMG_H, IMG_W, IMG_C
from fosae.train_fosae import PREFIX, TEMP_MIN
from fosae.train_fosae import MODEL_NAME as FOSAE_MODEL_NAME
from fosae.modules import STACKS, TRAIN_DATASETS_OBJS, MAX_N

N_EXAMPLES = 12

def init():

    train_set = Concat(
        [StateTransitionsDataset(
            hdf5_file="c_swm/data/blocks-{}-{}-{}_all.h5".format(OBJS, STACKS, 0), n_obj=OBJS + STACKS, max_n_obj=MAX_N
        ) for OBJS in TRAIN_DATASETS_OBJS]
    )
    print("Training Examples: {}".format(len(train_set)))

    view_loader = DataLoader(train_set, batch_size=N_EXAMPLES, shuffle=True)

    vae = FoSae().to(device)
    vae.load_state_dict(torch.load("fosae/model_{}/{}.pth".format(PREFIX, FOSAE_MODEL_NAME), map_location='cpu'))
    vae.eval()

    return vae, view_loader

def run(vae, view_loader):

    with torch.no_grad():

        data = view_loader.__iter__().__next__()

        _, _, obj_mask, next_obj_mask, obj_background, next_obj_background, action_mov_obj_index, action_from_obj_index, action_tar_obj_index,\
        state_n_obj, _, _, obj_mask_tilda, _, obj_background_tilda, _ = data

        data = obj_mask.to(device)
        data_next = next_obj_mask.to(device)
        data_tilda = obj_mask_tilda.to(device)
        state_n_obj = state_n_obj.to(device)

        back_grounds = torch.cat([obj_background, next_obj_background, obj_background_tilda], dim=1).to(device)

        # noise1 = torch.normal(mean=0, std=0.2, size=data.size()).to(device)
        # noise2 = torch.normal(mean=0, std=0.2, size=data_next.size()).to(device)
        # noise3 = torch.normal(mean=0, std=0.2, size=data_tilda.size()).to(device)

        action_idx = torch.cat([action_mov_obj_index, action_from_obj_index, action_tar_obj_index], dim=1).to(device)
        action_types = torch.zeros(size=(action_idx.size()[0],), dtype=torch.int32).to(device)
        action_n_obj = torch.tensor([3 for _ in range(action_idx.size()[0])]).to(device)
        action_input = (action_idx, action_n_obj, action_types)

        preds, change = vae(
            (data, data_next, data_tilda, state_n_obj, back_grounds), action_input,
            TEMP_MIN)
        preds, preds_next, _ = preds

        print(preds.size(), preds_next.size(), change.size())
        # print(preds)
        # print(preds_next)
        # exit(0)


        data_np = data.view(-1, MAX_N, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy().reshape(-1, MAX_N+1, MAX_N)

        print(data_np.shape, preds_np.shape)
        np.save("fosae/block_data/block_data.npy", data_np)
        np.save("fosae/block_data/block_preds.npy", preds_np)

        data_next_np = data_next.view(-1, MAX_N, IMG_C, IMG_H, IMG_W).detach().cpu().numpy()
        preds_next_np = preds_next.detach().cpu().numpy().reshape(-1, MAX_N+1, MAX_N)

        print(data_next_np.shape, preds_next_np.shape)
        np.save("fosae/block_data/block_data_next.npy", data_next_np)
        np.save("fosae/block_data/block_preds_next.npy", preds_next_np)

        change_np = change.detach().cpu().numpy().reshape(-1, MAX_N+1, MAX_N)
        print(change_np.shape)
        np.save("fosae/block_data/change.npy", change_np)



if __name__ == "__main__":
    vae, view_loader = init()
    run(vae, view_loader)
