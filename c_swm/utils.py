"""Utility functions."""

import os
import h5py
import numpy as np

import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt

EPS = 1e-17


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


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file, truncate=float('inf')):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file, truncate=truncate)

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0

        n_obj = np.unique(self.experience_buffer[0]['obs_obj_index']).shape[0]
        obj_mask_idx = self.experience_buffer[0]['obs_obj_index']
        next_obj_mask_idx = self.experience_buffer[0]['next_obj_index']
        assert obj_mask_idx.shape == next_obj_mask_idx.shape

        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action_one_hot'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

            obj_mask = np.zeros(shape=(n_obj, obj_mask_idx.shape[1], obj_mask_idx.shape[2]), dtype=np.float32)
            next_obj_mask = np.zeros(shape=(n_obj, next_obj_mask_idx.shape[1], next_obj_mask_idx.shape[2]), dtype=np.float32)

            # plt.gca()
            # plt.imshow(np.transpose(self.experience_buffer[ep]['obs'][0], (1,2,0)))
            # plt.pause(1)
            # plt.imshow(np.transpose(self.experience_buffer[ep]['next_obs'][0], (1,2,0)))
            # plt.pause(1)

            for i in range(n_obj):
                obj_mask[i] = (obj_mask_idx[0] == i).astype(np.float32)
                next_obj_mask[i] = (next_obj_mask_idx[0] == i).astype(np.float32)
                # print(obj_mask[i])
                # print(obj_mask[i].shape)
                # plt.imshow(obj_mask[i])
                # plt.pause(1)
                # plt.imshow(next_obj_mask[i])
                # plt.pause(1)
            # exit(0)

            self.experience_buffer[ep]['obj_mask'] = obj_mask
            self.experience_buffer[ep]['next_obj_mask'] = next_obj_mask








        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])

        action_one_hot = self.experience_buffer[ep]['action_one_hot'][step]

        obj_mask = self.experience_buffer[ep]['obj_mask']
        next_obj_mask = self.experience_buffer[ep]['next_obj_mask']

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

    dataset = StateTransitionsDataset(hdf5_file="data/blocks_eval.h5", truncate=50)
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
