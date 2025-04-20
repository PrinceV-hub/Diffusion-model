import matplotlib.pyplot as plt
import torch
import random

# Choose a random sample image from the dataset
random_index = random.randint(0, len(dataset) - 1)  # Select a random index
sample_img, label = dataset[random_index]  # Get image and label
label_name = dataset.classes[label]  # Get the label name corresponding to the label index

# Add noise to the sample image using forward diffusion at timestep T (maximum noise)
x_t, _ = forward_diffusion_sample(sample_img.unsqueeze(0), torch.tensor([T-1]).to(device))

# Timesteps to visualize how image is denoised
timesteps = [T-1, 800, 600, 400, 200, 100, 50, 3, 2, 0]  # Different timesteps for denoising

# Create a figure to plot images at different timesteps (2 rows, 6 columns)
fig, axs = plt.subplots(2, 6, figsize=(18, 8))  # 2 rows and 6 columns to include original image

# Display the original image with its label
original_img = (sample_img + 1) / 2  # Rescale to [0, 1] for display
axs[0, 0].imshow(original_img.permute(1, 2, 0))  # Convert CxHxW to HxWxC for display
axs[0, 0].set_title(f"Original Image\nLabel: {label_name}")
axs[0, 0].axis('off')

# Show noisy image at the final timestep (maximum noise)
noisy_img = (x_t.squeeze() + 1) / 2  # Rescale from [-1, 1] to [0, 1]
noisy_img = torch.clamp(noisy_img, 0, 1)  # Ensure the image is in the correct range
axs[0, 1].imshow(noisy_img.permute(1, 2, 0))  # Convert CxHxW to HxWxC for display
axs[0, 1].set_title(f"Noisy Image at T={T-1}\nLabel: {label_name}")
axs[0, 1].axis('off')

# Show denoised images at different timesteps
for i, t in enumerate(timesteps):
    # Reverse diffusion step: denoise at timestep `t`
    x_t, _ = forward_diffusion_sample(sample_img.unsqueeze(0), torch.tensor([t]).to(device))

    denoised_img = (x_t.squeeze() + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    denoised_img = torch.clamp(denoised_img, 0, 1)  # Clip values to [0, 1] to avoid clipping warning

    # Adjusting the grid position for each timestep
    row = (i + 2) // 6  # Place images starting from row 1
    col = (i + 2) % 6  # Column in the grid
    
    ax = axs[row, col]
    ax.imshow(denoised_img.permute(1, 2, 0))  # Convert CxHxW to HxWxC for display
    ax.set_title(f"Timestep {t}\nLabel: {label_name}")
    ax.axis('off')

plt.tight_layout()
plt.show()
