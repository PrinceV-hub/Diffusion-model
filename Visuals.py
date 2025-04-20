import matplotlib.pyplot as plt

# Class labels for CIFAR-10
classes = dataset.classes  # ['airplane', 'automobile', ..., 'truck']

# Visualize a few random images from the dataset
fig, axs = plt.subplots(2, 5, figsize=(6, 6))  # 2x5 grid for images
for i in range(10):
    img, label = dataset[i]  # Get image and label from dataset
    img = (img + 1) / 2  # Rescale from [-1, 1] to [0, 1] for visualization
    axs[i // 5, i % 5].imshow(img.permute(1, 2, 0))  # Convert CxHxW to HxWxC for display
    axs[i // 5, i % 5].set_title(classes[label])  # Set title as the class label
    axs[i // 5, i % 5].axis('off')  # Hide axes

plt.tight_layout()
plt.show()
