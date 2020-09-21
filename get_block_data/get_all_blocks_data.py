from fosae.modules import STACKS, REMOVE_BG
import torch
from torch.utils.data import DataLoader
from c_swm.utils import StateTransitionsDataset
import pickle
import sys

if len(sys.argv) == 1:
    print("Please Specify for 'all' or 'half' for dataset size")
    exit(0)
else:
    if sys.argv[1] == "all":
        N_OBJS = [1, 2, 3, 4]
    else:
        assert sys.argv[1] == "half"
        N_OBJS = [1, 2]


all_obs = []
all_next_obs = []
all_obj_mask = []
all_next_obj_mask = []
all_obs_tilda = []
all_next_obs_tilda = []
all_obj_mask_tilda = []
all_next_obj_mask_tilda = []
all_action_mov_obj_index = []
all_action_tar_obj_index = []
all_n_obj = []

BATCH_SIZE = 200


for OBJS in N_OBJS:
    print(OBJS)
    dataset = StateTransitionsDataset(hdf5_file="c_swm/data/blocks-{}-{}-det_all.h5".format(OBJS, STACKS),
                                      n_obj=OBJS + STACKS, remove_bg=False, max_n_obj=9)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for batch_idx, data_batch in enumerate(dataloader):
        obs, next_obs, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index,\
            obs_tilda, next_obs_tilda, obj_mask_tilda, next_obj_mask_tilda = data_batch
        all_obs.append(obs)
        all_next_obs.append(next_obs)
        all_obj_mask.append(obj_mask)
        all_next_obj_mask.append(next_obj_mask)
        all_action_mov_obj_index.append(action_mov_obj_index)
        all_action_tar_obj_index.append(action_tar_obj_index)
        all_obs_tilda.append(obs_tilda)
        all_next_obs_tilda.append(next_obs_tilda)
        all_obj_mask_tilda.append(obj_mask_tilda)
        all_next_obj_mask_tilda.append(next_obj_mask_tilda)
        all_n_obj.append(torch.tensor(OBJS+STACKS).unsqueeze(0).expand(obs.size(0), -1))

all_obs = torch.cat(all_obs)
all_next_obs = torch.cat(all_next_obs)
all_obj_mask = torch.cat(all_obj_mask)
all_next_obj_mask = torch.cat(all_next_obj_mask)
all_action_mov_obj_index = torch.cat(all_action_mov_obj_index)
all_action_tar_obj_index = torch.cat(all_action_tar_obj_index)
all_obs_tilda = torch.cat(all_obs_tilda)
all_next_obs_tilda = torch.cat(all_next_obs_tilda)
all_obj_mask_tilda = torch.cat(all_obj_mask_tilda)
all_next_obj_mask_tilda = torch.cat(all_next_obj_mask_tilda)
all_n_obj = torch.cat(all_n_obj)

print(all_obs.size())
print(all_next_obs.size())
print(all_obj_mask.size())
print(all_next_obj_mask.size())
print(all_action_mov_obj_index.size())
print(all_action_tar_obj_index.size())
print(all_obs_tilda.size())
print(all_next_obs_tilda.size())
print(all_obj_mask_tilda.size())
print(all_next_obj_mask_tilda.size())
print(all_n_obj.size())

pickle.dump(
    {
        'obs': all_obs, 'next_obs': all_next_obs, 'obj_mask': all_obj_mask, 'next_obj_mask': all_next_obj_mask,
        'action_mov_obj_index': all_action_mov_obj_index, 'action_tar_obj_index': all_action_tar_obj_index,
        'obs_tilda': all_obs_tilda, 'next_obs_tilda': all_next_obs_tilda,
        'obj_mask_tilda': all_obj_mask_tilda, 'next_obj_mask_tilda': all_next_obj_mask_tilda,
        'n_obj': all_n_obj
    },
    open("c_swm/data/blocks-{}-size-det_all.pkl".format(sys.argv[1]), 'w'), protocol=4
)
