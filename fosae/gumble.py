import torch

GUMBLE_NOISE = True
if torch.cuda.is_available():
    device = 'cuda:0'
    print("Using GPU")
else:
    device = 'cpu'
    print("Using CPU")

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, add_noise=GUMBLE_NOISE):
    if logits.size()[-1] == 1:
        if temperature == 0: # make it to sigmoid if it is a bit choice
            ret = (logits > 0.5).float()
        else:
            if add_noise:
                noise1 = sample_gumbel(logits.size())
                noise2 = sample_gumbel(logits.size())
                logits = (logits + noise1) / (1 + noise2)
            ret = torch.sigmoid(logits / temperature)
    else:
        if temperature == 0: # not differentiable in test case, but it is ok
            _, ind = torch.max(logits, dim=-1)
            ret = torch.zeros(size=logits.size()).to(device).scatter(dim=-1, index=ind.unsqueeze(-1), value=1)
        else:
            if add_noise:
                noise = sample_gumbel(logits.size())
                logits = logits + noise
            ret = torch.softmax(logits / temperature, dim=-1)
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