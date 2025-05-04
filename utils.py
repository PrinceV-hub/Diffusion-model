from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),  # Scale to [-1, 1]
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader
