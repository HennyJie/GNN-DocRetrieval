import random
import torch
import torch.nn
import numpy as np
import os

def get_activation(name: str):
    activations = {
        'relu': torch.nn.ReLU,
        'hardtanh': torch.nn.Hardtanh,
        'elu': torch.nn.ELU,
        'leakyrelu': torch.nn.LeakyReLU,
        'prelu': torch.nn.PReLU,
        'rrelu': torch.nn.RReLU
    }

    return activations[name]


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)