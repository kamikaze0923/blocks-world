import os
from symbolic_rep.block import Block, prefix, extract_predicate, objs
import matplotlib.pyplot as plt
import numpy as np
from c_swm.utils import save_list_dict_h5py
from skimage.transform import resize

NUM_EPISODE = 100
EPISODE_LENGTH = 10
N_TRAIN = 5000
N_EVAL = 760

def set_image_action(obs, obj_color):
    target = np.tile(obj_color, (obs.shape[0], obs.shape[1], 1))
    cond = np.all(obs == target, axis=-1).astype(np.int32)
    # rgb = []
    # for _ in range(3):
    #     rgb.append(cond)
    # rgb.append(np.ones(shape=cond.shape))
    # rgb = np.stack(rgb, axis=-1)
    # plt.imshow(rgb)
    # plt.show()
    return cond

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
        if i < N_TRAIN:
            replay_buffer = replay_buffer_train
        else:
            replay_buffer = replay_buffer_eval
        pre_json = os.path.join(prefix, "scene_tr", tr_files[i*2])
        suc_json = os.path.join(prefix, "scene_tr", tr_files[i*2+1])
        pre_objs, bottom_pads, _ = extract_predicate(pre_json)
        suc_objs, _, _ = extract_predicate(suc_json)
        action = None
        target_obj = None
        moving_obj = None
        for pre_obj, suc_obj in zip(pre_objs, suc_objs):
            assert pre_obj.id == suc_obj.id
            assert isinstance(pre_obj, Block)
            if pre_obj.position_eq(suc_obj):
                # print("Block {} does not move".format(pre_obj.id))
                pass
            else:
                # print("Pick up block {} from stack {},  Drop it on stack {}".format(pre_obj.id, pre_obj.n_stack, suc_obj.n_stack))
                # print("Moving object {}".format(pre_obj))
                moving_obj = pre_obj
                if suc_obj.floor == 0:
                    target_obj = bottom_pads[suc_obj.n_stack]
                else:
                    for obj in suc_objs:
                        if obj.floor == suc_obj.floor - 1:
                            target_obj = obj
                            break
                # print("Target object {}".format(target_obj))
                action = ACTIONS.index((pre_obj.n_stack, suc_obj.n_stack))
                break
        assert action is not None
        assert target_obj is not None
        assert moving_obj is not None
        replay = {
            'obs': [],
            'action_one_hot': [],
            'action_image': [],
            'next_obs': [],
        }

        replay['action_one_hot'].append(action)
        obs = plt.imread(os.path.join(prefix, "mask_image_tr", img_tr_files[i*2]))
        next_obs = plt.imread(os.path.join(prefix, "mask_image_tr", img_tr_files[i*2+1]))
        obs_colors = np.unique(np.resize(obs, (-1, 4)), axis=0)
        next_obs_colors = np.unique(np.resize(next_obs, (-1, 4)), axis=0)
        assert obs_colors.shape[0] == 9
        assert next_obs_colors.shape[0] == 9
        action_mov_obs = set_image_action(obs, moving_obj.color)
        action_tar_obs = set_image_action(obs, target_obj.color)
        action_image = np.stack([action_mov_obs, action_tar_obs])
        replay['action_image'].append(
            resize(action_image, (2, 100, 150))
        )
        # plt.imshow(obs)
        # plt.show()
        # plt.imshow(next_obs)
        # plt.show()
        replay['obs'].append(
            resize(np.transpose(obs, (2,0,1)), (4, 100, 150))
        )

        replay['next_obs'].append(
            resize(np.transpose(next_obs, (2,0,1)), (4, 100, 150))
        )

        replay_buffer.append(replay)

    save_list_dict_h5py(replay_buffer_train, "{}/{}/{}".format("c_swm", "data", "blocks_train.h5"))
    save_list_dict_h5py(replay_buffer_eval, "{}/{}/{}".format("c_swm", "data", "blocks_eval.h5"))


if __name__ == "__main__":
    gen_episode(NUM_EPISODE, EPISODE_LENGTH)






