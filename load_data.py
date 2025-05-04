# STEP 2: Load CIFAR-10 Dataset
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

# Visualize a few random images
classes = dataset.classes
fig, axs = plt.subplots(2, 5, figsize=(6, 6))
for i in range(10):
    img, label = dataset[i]
    img = (img + 1) / 2
    axs[i // 5, i % 5].imshow(img.permute(1, 2, 0))
    axs[i // 5, i % 5].set_title(classes[label])
    axs[i // 5, i % 5].axis('off')

plt.tight_layout()
plt.show()
