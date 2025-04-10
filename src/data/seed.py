"""Ensures consistent / reproducable shuffling in datasets"""

import torch

seed = 42
generator = torch.Generator()
generator.manual_seed(seed)
