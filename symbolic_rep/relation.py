class SceneRelation:
    def __init__(self):
        self.on_ground = set()
        self.on_block = {}
        self.clear = set()

    def print_relation(self):
        print(self.on_ground)
        print(self.on_block)
        print(self.clear)