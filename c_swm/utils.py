"""Utility functions."""

import os
import h5py
import numpy as np

import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt


EPS = 1e-17

np.set_printoptions(linewidth=np.inf, threshold=np.inf)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py(fname, truncate=float('inf')):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
            if i ==truncate:
                break
    return array_dict


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def pairwise_distance_matrix(x, y):
    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32, device=indices.device
    )
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result

def get_masked_obj(obs, idx, idx_matirx):
    flt = np.stack([(idx_matirx[0] == idx).astype(np.float32) for _ in range(obs.shape[1])])
    prod = np.multiply(flt, obs[0])
    prod[prod == 0] = 1
    return prod



class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file, n_obj, truncate=float('inf')):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file, truncate=truncate)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0

        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action_one_hot'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

            obs = self.experience_buffer[ep]['obs']
            next_obs = self.experience_buffer[ep]['next_obs']
            obj_mask_idx = self.experience_buffer[ep]['obs_obj_index']
            next_obj_mask_idx = self.experience_buffer[ep]['next_obj_index']
            assert obj_mask_idx.shape == next_obj_mask_idx.shape

            obj_mask_sep = np.zeros(shape=(n_obj, obs.shape[1], obs.shape[2], obs.shape[3]), dtype=np.float32)
            next_obj_mask_sep = np.zeros(shape=(n_obj, next_obs.shape[1], next_obs.shape[2], next_obs.shape[3]), dtype=np.float32)

            # plt.gca()
            # plt.imshow(np.transpose(self.experience_buffer[ep]['obs'][0], (1,2,0)))
            # plt.pause(1)
            # plt.imshow(np.transpose(self.experience_buffer[ep]['next_obs'][0], (1,2,0)))
            # plt.pause(1)

            for i in range(n_obj):
                obj_mask_sep[i] = get_masked_obj(obs, i, obj_mask_idx)
                next_obj_mask_sep[i] = get_masked_obj(next_obs, i, next_obj_mask_idx)
            #     plt.imshow(np.transpose(obj_mask_sep[i], (1,2,0)))
            #     plt.pause(1)
            #     plt.imshow(np.transpose(next_obj_mask_sep[i], (1,2,0)))
            #     plt.pause(1)
            # exit(0)

            self.experience_buffer[ep]['obj_mask_sep'] = obj_mask_sep
            self.experience_buffer[ep]['next_obj_mask_sep'] = next_obj_mask_sep

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])

        action_one_hot = self.experience_buffer[ep]['action_one_hot'][step]

        obj_mask = self.experience_buffer[ep]['obj_mask_sep']
        next_obj_mask = self.experience_buffer[ep]['next_obj_mask_sep']

        action_mov_obj_index = self.experience_buffer[ep]['action_mov_obj_index']
        action_tar_obj_index = self.experience_buffer[ep]['action_tar_obj_index']

        return obs, next_obs, action_one_hot, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index




class PathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, action_encoding, path_length=5, truncate=float('inf')):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file, truncate=truncate)
        self.path_length = path_length
        self.action_type = action_encoding

    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):
        observations = []
        actions = []
        for i in range(self.path_length):
            obs = to_float(self.experience_buffer[idx]['obs'][i])
            action = self.experience_buffer[idx][self.action_type][i]
            observations.append(obs)
            actions.append(action)
        obs = to_float(
            self.experience_buffer[idx]['next_obs'][self.path_length - 1])
        observations.append(obs)
        return observations, actions



if __name__ == "__main__":

    dataset = StateTransitionsDataset(hdf5_file="data/blocks_eval.h5", n_obj=9, truncate=50)
    eval_loader = data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

    for batch_idx, data_batch in enumerate(eval_loader):
        data_batch = [tensor.to('cpu') for tensor in data_batch]
        obs, next_obs, action_one_hot, obj_mask, next_obj_mask, action_mov_obj_index, action_tar_obj_index = data_batch
        print(obs.size())
        print(next_obs.size())
        print(action_one_hot.size())
        print(obj_mask.size())
        print(next_obj_mask.size())
        print(action_mov_obj_index.size())
        print(action_tar_obj_index.size())
        print(obj_mask.dtype)
        exit(0)
