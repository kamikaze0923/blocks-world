import os
from symbolic_rep.block import Block, prefix, extract_predicate, objs
import numpy as np

tr_files = os.listdir(os.path.join(prefix, "scene_tr"))
tr_files.sort()
n_tr = len(tr_files) // 2

ACTIONS = [(i,j) for i in range(objs) for j in range(objs) if j != i] # (SOURCE, TARGET)


actions = []
for i in range(0, len(tr_files), 2):
    pre_json = os.path.join(prefix, "scene_tr", tr_files[i])
    suc_json = os.path.join(prefix, "scene_tr", tr_files[i+1])
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
    actions.append(action)

actions = np.array(actions)
print(actions.shape, actions.dtype)
np.save("{}/{}-actions.npy".format(prefix, prefix), actions)






