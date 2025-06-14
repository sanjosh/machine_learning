import random

import numpy as np
import torch

def get_study_number():
    return 9

def get_study_name(number = None):
    if not number:
        number = get_study_number()
    return f"traffic{number}"

HOURLY_LEN = 168
FIVEMIN_LEN = 72
DIM_HOURLY = 8
DIM_5MIN = 8
# increasing hourly len and fivemin len can increase loss,
# remedy by increasing model capacity (d_model, n_heads, n_layers)
# For longer sequences, use learned or rotary positional embeddings rather than sinusoidal ones.
# MSE Loss scales with sequence length
# normalize loss by sequence length
# loss_5min = criterion_5min(pred_5min, y_5min) / fivemin_seq.size(1)
# or use MSELoss(reduction='mean')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False