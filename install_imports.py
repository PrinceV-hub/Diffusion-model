# STEP 1: Install & Import
!pip install -q torch torchvision matplotlib tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

device = "cpu"  # forced CPU for Colab
