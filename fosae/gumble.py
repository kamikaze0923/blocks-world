import torch
from torch.nn import functional as F

if torch.cuda.is_available():
    device = 'cuda:0'
    print("Using GPU")
else:
    device = 'cpu'
    print("Using CPU")

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    if temperature == 0: # not differentiable in test case, but it is ok
        _, ind = torch.max(logits, dim=-1)
        ret = torch.zeros(size=logits.size()).to(device).scatter(dim=-1, index=ind.unsqueeze(-1), value=1)
    else:
        noise = sample_gumbel(logits.size())
        y = logits + noise
        ret = F.softmax(y / temperature, dim=-1)
    return ret

def gumbel_softmax(q_y, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """

    y = gumbel_softmax_sample(q_y, temperature)
    if hard:
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros(size=y.size()).to(device).scatter(dim=-1, index=ind.unsqueeze(-1), value=1)
        y_ret = (y_hard - y).detach() + y
    else:
        y_ret = y
    return y_ret