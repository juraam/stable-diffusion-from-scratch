import random, os
import numpy as np
import torch

def seed_everything(seed):   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if seed is None:
        torch.manual_seed(torch.initial_seed())
        torch.cuda.manual_seed(torch.initial_seed())
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True