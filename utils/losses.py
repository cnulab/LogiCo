import torch
from functools import partial


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x


def global_cosine_hm_adaptive(a_, b_, y=2):
    cos_loss = torch.nn.CosineSimilarity()

    with torch.no_grad():
         point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()
    mean_dist = point_dist.mean()

    factor = (point_dist / mean_dist) ** (y)
    loss = torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                   b_.reshape(b_.shape[0], -1)))

    partial_func = partial(modify_grad_v2, factor=factor)
    b_.register_hook(partial_func)
    return loss

