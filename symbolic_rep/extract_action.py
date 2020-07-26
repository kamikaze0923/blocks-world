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

def gen_episode(num_episode, episode_length):

    tr_files = os.listdir(os.path.join(prefix, "scene_tr"))
    tr_files.sort()
    n_tr = len(tr_files) // 2
    assert N_TRAIN + N_EVAL == n_tr

    img_tr_files = os.listdir(os.path.join(prefix, "image_tr"))
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
        pre_objs, pre_relations = extract_predicate(pre_json)
        suc_objs, suc_relations = extract_predicate(suc_json)
        action = None
        for pre_obj, suc_obj in zip(pre_objs, suc_objs):
            assert pre_obj.id == suc_obj.id
            assert isinstance(pre_obj, Block)
            if pre_obj.position_eq(suc_obj):
                # print("Block {} does not move".format(pre_obj.id))
                pass
            else:
                # print("Pick up block {} from stack {},  Drop it on stack {}".format(pre_obj.id, pre_obj.n_stack, suc_obj.n_stack))
                action = ACTIONS.index((pre_obj.n_stack, suc_obj.n_stack))
        assert action is not None

        replay = {
            'obs': [],
            'action': [],
            'next_obs': [],
        }

        replay['action'].append(action)
        replay['obs'].append(
            resize(np.transpose(plt.imread(os.path.join(prefix, "mask_image_tr", img_tr_files[i*2])), (2,0,1)), (4, 100, 150))
        )
        # plt.imshow(resize(np.transpose(plt.imread(os.path.join(prefix, "image_tr", img_tr_files[i*2])), (2,0,1)), (4, 100, 150)).transpose(1,2,0))
        # plt.show()
        replay['next_obs'].append(
            resize(np.transpose(plt.imread(os.path.join(prefix, "mask_image_tr", img_tr_files[i*2+1])), (2,0,1)), (4, 100, 150))
        )

        replay_buffer.append(replay)

    save_list_dict_h5py(replay_buffer_train, "{}/{}/{}".format("c_swm", "data", "blocks_train.h5"))
    save_list_dict_h5py(replay_buffer_eval, "{}/{}/{}".format("c_swm", "data", "blocks_eval.h5"))


if __name__ == "__main__":
    gen_episode(NUM_EPISODE, EPISODE_LENGTH)






