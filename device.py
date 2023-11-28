import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#device = 'cpu'

if device == 'cpu':
    num_workers = 0
else:
    num_workers = 0