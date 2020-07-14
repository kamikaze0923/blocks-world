import os
from symbolic_rep.block import Block, prefix, extract_predicate, objs
import matplotlib.pyplot as plt
import numpy as np
from c_swm.utils import save_list_dict_h5py

NUM_EPISODE = 100
EPISODE_LENGTH = 10

def gen_episode(num_episode, episode_length, save_file):

    tr_files = os.listdir(os.path.join(prefix, "scene_tr"))
    tr_files.sort()
    n_tr = len(tr_files) // 2

    img_tr_files = os.listdir(os.path.join(prefix, "image_tr"))
    img_tr_files.sort()

    ACTIONS = [(i, j) for i in range(objs) for j in range(objs) if j != i]  # (SOURCE, TARGET)

    replay_buffer = []

    for i in range(n_tr):
        if i % 100 == 0:
            print(i)
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

        replay_buffer.append({
            'obs': [],
            'action': [],
            'next_obs': [],
        })

        replay_buffer[i]['action'].append(action)
        replay_buffer[i]['obs'].append(
            np.transpose(plt.imread(os.path.join(prefix, "image_tr", img_tr_files[i*2])), (2,0,1))
        )
        replay_buffer[i]['next_obs'].append(
            np.transpose(plt.imread(os.path.join(prefix, "image_tr", img_tr_files[i*2+1])), (2,0,1))
        )


    fname = "{}/{}/{}".format("c_swm", "data", save_file)
    save_list_dict_h5py(replay_buffer, fname)



if __name__ == "__main__":
    save_file = "block_train.h5"
    gen_episode(NUM_EPISODE, EPISODE_LENGTH, save_file)






