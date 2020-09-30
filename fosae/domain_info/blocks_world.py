import torch
from fosae.gumble import device

STACKS = 4
TRAIN_DATASETS_OBJS = [1]
MAX_N = max(TRAIN_DATASETS_OBJS) + STACKS

Ps = [1,1] # how may predicates learned for each arity
As = [3] # how many arity for each arity

class MoveAction:

    CLEAR = torch.cartesian_prod(torch.arange(MAX_N)).to(device)
    ON = torch.cartesian_prod(torch.arange(MAX_N), torch.arange(MAX_N)).to(device)

    def get_precondition(self, moving_obj, from_obj, target_obj):

        """
        (clear ?b1),
        (clear ?b3),
        (on ?b1 ?b2)
        """

        clear_mov = torch.argmax((self.CLEAR == moving_obj).int())
        clear_tar = torch.argmax((self.CLEAR == target_obj).int())
        on_mov_from = torch.argmax((self.ON == torch.stack([moving_obj, from_obj])).all(dim=1).int())

        return torch.tensor([clear_mov, clear_tar, on_mov_from + MAX_N]), torch.tensor([1,1,1]).float()

    def get_effect(self, moving_obj, from_obj, target_obj):

        """
        (clear ?b2),
        (not (clear ?b3)),
        (on ?b1 ?b3),
        (not (on ?b1 ?b2))
        """

        clear_from = torch.argmax((self.CLEAR == from_obj).int())
        not_clear_tar = torch.argmax((self.CLEAR == target_obj).int())
        on_mov_tar = torch.argmax((self.ON == torch.stack([moving_obj, target_obj])).all(dim=1).int())
        not_on_mov_from = torch.argmax((self.ON == torch.stack([moving_obj, from_obj])).all(dim=1).int())

        return torch.tensor([clear_from, not_clear_tar, on_mov_tar + MAX_N, not_on_mov_from + MAX_N]), torch.tensor([1,0,1,0]).float()


ACTION_FUNC = [MoveAction()]
assert len(As) == len(ACTION_FUNC)