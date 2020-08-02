import os
from symbolic_rep.block import Block, prefix, extract_predicate, objs
import matplotlib.pyplot as plt
import numpy as np
from c_swm.utils import save_list_dict_h5py


NUM_EPISODE = 100
EPISODE_LENGTH = 10
N_TRAIN = 5000
N_EVAL = 760

np.random.seed(0)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
TRAIN_IDX = np.random.choice(N_TRAIN+N_EVAL, N_TRAIN, replace=False)

def resize(img): # resize by 2
    assert len(img.shape) == 3 or len(img.shape) == 2
    if len(img.shape) == 3:
        img_down_sampled = img[:,::2,::2]
    else:
        img_down_sampled = img[::2, ::2]
    return img_down_sampled

def get_index_from_image(img, color):
    target = np.tile(color, (img.shape[0], img.shape[1], 1))
    cond = np.all(img == target, axis=-1)
    return np.where(cond)

def gen_episode(num_episode, episode_length):

    tr_files = os.listdir(os.path.join(prefix, "scene_tr"))
    tr_files.sort()
    n_tr = len(tr_files) // 2
    assert N_TRAIN + N_EVAL == n_tr

    img_tr_files = os.listdir(os.path.join(prefix, "mask_image_tr"))
    img_tr_files.sort()

    ACTIONS = [(i, j) for i in range(objs) for j in range(objs) if j != i]  # (SOURCE, TARGET)

    replay_buffer_train = []
    replay_buffer_eval = []

    for i in range(n_tr):
        if i % 100 == 0:
            print(i)
        if i in TRAIN_IDX:
            replay_buffer = replay_buffer_train
        else:
            replay_buffer = replay_buffer_eval
        pre_json = os.path.join(prefix, "scene_tr", tr_files[i*2])
        suc_json = os.path.join(prefix, "scene_tr", tr_files[i*2+1])
        pre_objs, pre_bottom_pads, _ = extract_predicate(pre_json)
        suc_objs, suc_bottom_pads, _ = extract_predicate(suc_json)
        action = None
        target_obj = None
        moving_obj = None

        obs = plt.imread(os.path.join(prefix, "mask_image_tr", img_tr_files[i*2]))
        next_obs = plt.imread(os.path.join(prefix, "mask_image_tr", img_tr_files[i*2+1]))

        pre_objs_index_matrix = np.zeros(shape=obs.shape[:2])
        next_objs_index_matrix = np.zeros(shape=next_obs.shape[:2])

        assert len(pre_objs) == len(suc_objs)
        for pre_obj, suc_obj in zip(pre_objs, suc_objs):
            assert pre_obj.id == suc_obj.id
            assert isinstance(pre_obj, Block)
            pre_obj_index = get_index_from_image(obs, pre_obj.color)
            next_obj_index = get_index_from_image(next_obs, suc_obj.color)
            pre_objs_index_matrix[pre_obj_index[0], pre_obj_index[1]] = pre_obj.id #0 as back ground
            next_objs_index_matrix[next_obj_index[0], next_obj_index[1]] = suc_obj.id #0 as back ground
            if pre_obj.position_eq(suc_obj):
                # print("Block {} does not move".format(pre_obj.id))
                pass
            else:
                # print("Pick up block {} from stack {},  Drop it on stack {}".format(pre_obj.id, pre_obj.n_stack, suc_obj.n_stack))
                # print("Moving object {}".format(pre_obj))
                moving_obj = pre_obj
                if suc_obj.floor == 0:
                    target_obj = pre_bottom_pads[suc_obj.n_stack]
                else:
                    for obj in suc_objs:
                        if obj.floor == suc_obj.floor - 1 and obj.n_stack == suc_obj.n_stack:
                            target_obj = obj
                            break
                # print("Target object {}".format(target_obj))
                action = ACTIONS.index((pre_obj.n_stack, suc_obj.n_stack))
                break

        for (pre_pad, suc_pad) in zip(pre_bottom_pads, suc_bottom_pads):
            pre_obj_index = get_index_from_image(obs, pre_pad.color)
            pre_objs_index_matrix[pre_obj_index[0], pre_obj_index[1]] = pre_pad.id
            suc_obj_index = get_index_from_image(obs, suc_pad.color)
            next_objs_index_matrix[suc_obj_index[0], suc_obj_index[1]] = suc_pad.id

        assert action is not None
        assert target_obj is not None
        assert moving_obj is not None
        replay = {
            'action_one_hot': [],
            'obs': [],
            'obs_obj_index': [],
            'next_obs': [],
            'next_obj_index': [],
            'action_mov_obj_index': [],
            'action_tar_obj_index': []
        }

        replay['action_one_hot'].append(action)

        obs_colors = np.unique(np.resize(obs, (-1, 4)), axis=0)
        next_obs_colors = np.unique(np.resize(next_obs, (-1, 4)), axis=0)
        assert obs_colors.shape[0] == 9
        assert next_obs_colors.shape[0] == 9

        # action_mov_obs = set_image_action(obs, moving_obj.color)
        # action_tar_obs = set_image_action(obs, target_obj.color)
        # action_image = np.stack([action_mov_obs, action_tar_obs]).astype(np.float32)
        # replay['action_image'].append(
        #     resize(action_image, (2, 100, 150))
        # )
        # plt.imshow(obs)
        # plt.show()
        # plt.imshow(next_obs)
        # plt.show()
        replay['obs'].append(
            resize(np.transpose(obs, (2,0,1)))
        )

        replay['next_obs'].append(
            resize(np.transpose(next_obs, (2,0,1)))
        )

        replay['obs_obj_index'].append(
            resize(pre_objs_index_matrix)
        )

        replay['next_obj_index'].append(
            resize(next_objs_index_matrix)
        )

        replay['action_mov_obj_index'].append(
            moving_obj.id
        )

        replay['action_tar_obj_index'].append(
            target_obj.id
        )

        replay_buffer.append(replay)

    save_list_dict_h5py(replay_buffer_train, "{}/{}/{}".format("c_swm", "data", "blocks_train.h5"))
    save_list_dict_h5py(replay_buffer_eval, "{}/{}/{}".format("c_swm", "data", "blocks_eval.h5"))


if __name__ == "__main__":
    gen_episode(NUM_EPISODE, EPISODE_LENGTH)






