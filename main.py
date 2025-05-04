from model import SimpleUNet
from utils import get_data_loader
from train import train
import torch

device = "cpu"
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataset, loader = get_data_loader()
train(model, optimizer, loader, device=device)
