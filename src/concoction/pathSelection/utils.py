import torch
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sparse_colmun(x):
    # L2,1
    x = x * x
    x = torch.sum(x, 0)
    x = torch.sqrt(x)
    return torch.sum(x)
