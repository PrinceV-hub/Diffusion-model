transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1),  # Scale to [-1, 1]
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Total images: {len(dataset)}")

# Check one sample
sample_img, label = dataset[0]
print(f"Single image shape: {sample_img.shape}")
print(f"Pixel value range: min {sample_img.min():.2f}, max {sample_img.max():.2f}")
