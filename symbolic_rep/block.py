import os
import json
from collections import defaultdict
from symbolic_rep.relation import SceneRelation

class Block:

    BLCOK_NAMES = ["A", "B", "C", "D"]

    def __init__(self, object_config):
        self.shape = object_config['shape']
        self.size = object_config['size']
        self.color = object_config['color']
        if 'location' in object_config:
            self.n_stack = STACK_XS.index(object_config['location'][0])
            self.z = object_config['location'][2]

    def __eq__(self, other):
        assert isinstance(other, Block)
        if self.shape != other.shape:
            return False
        if self.size != other.size:
            return False
        for c1, c2 in zip(self.color, other.color):
            if c1 != c2:
                return False
        return True

    def __repr__(self):
        return "\nshape: {} size:{: .2f} color:({:.2f}, {:.2f}, {:.2f})".format(
            self.shape, self.size, self.color[0], self.color[1], self.color[2]
        )

    def set_id(self, id):
        self.id = self.BLCOK_NAMES[id]

    def set_floor(self, floor):
        self.floor = floor

    def position_eq(self, other):
        return self.n_stack == other.n_stack and self.floor == other.floor

    def print_block_scene_position(self):
        print("Block on {}th stack and {}th floor".format(self.n_stack, self.floor))

def extract_predicate(json_file):
    # print(json_file)
    with open(json_file) as f:
        state_json = json.load(f)
    scene_stacks_ys = defaultdict(lambda: [])
    scene_objs = []
    relation = SceneRelation()
    for obj in state_json['objects']:
        b = Block(obj)
        scene_objs.append(b)
        if b in SCENE_OBJS:
            b.set_id(SCENE_OBJS.index(b))
        scene_stacks_ys[b.n_stack].append(b.z)
    for k, v in scene_stacks_ys.items():
        scene_stacks_ys[k] = sorted(v)
    for b in scene_objs:
        b.set_floor(scene_stacks_ys[b.n_stack].index(b.z))
        # b.print_block_scene_position()
    for b in scene_objs:
        clear = True
        if b.floor == 0:
            relation.on_ground.add(b.id)
        for other_b in scene_objs:
            if other_b.floor == b.floor - 1:
                relation.on_block[b.id] = other_b.id
            elif other_b.floor > b.floor:
                clear = False
        if clear:
            relation.clear.add(b.id)
    return scene_objs, relation

objs = 4
stacks = 4
det = True
prefix = "blocks-{}-{}-{}".format(objs, stacks, "det" if det else "")
print(prefix)

with open((os.path.join(prefix, "{}-init.json".format(prefix)))) as f:
    init_json = json.load(f)

STACK_XS = init_json['stack_x']
SCENE_OBJS = [Block(config) for config in init_json['objects']]
