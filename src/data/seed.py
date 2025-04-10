"""Ensures consistent / reproducable shuffling in datasets"""

import torch

seed = 42
g = torch.Generator()
g.manual_seed(seed)
