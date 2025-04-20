# Choose a sample image from the dataset
import random

# Choose a random index from the dataset
random_index = random.randint(0, len(dataset) - 1)  # Select a random index between 0 and len(dataset)-1

# Get the image and label at the random index
sample_img, label = dataset[random_index]  # Get the image and label at the random index
label_name = dataset.classes[label]  # Get the label name corresponding to the label index

# Now you can proceed with visualizing or processing the sample image
print(f"Random Image Label: {label_name}")

# Plot the image at different timesteps
timesteps = [0, 200, 400, 600, 800, T-1]  # Timesteps to visualize

# Create a figure to plot images at different timesteps
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Display the original image with its label
original_img = (sample_img + 1) / 2  # Rescale to [0, 1] for display
axs[0, 0].imshow(original_img.permute(1, 2, 0))  # Convert CxHxW to HxWxC for display
axs[0, 0].set_title(f"Original Image\nLabel: {label_name}")
axs[0, 0].axis('off')

# Show images after adding noise at different timesteps
for i, t in enumerate(timesteps):
    x_t, _ = forward_diffusion_sample(sample_img.unsqueeze(0), torch.tensor([t]).to(device))  # Add noise at timestep `t`

    noisy_img = (x_t.squeeze() + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    noisy_img = torch.clamp(noisy_img, 0, 1)  # Clip values to [0, 1] to avoid clipping warning

    # Plot each noisy image at the corresponding timestep
    ax = axs[i // 3, i % 3]  # Position on grid for each timestep
    ax.imshow(noisy_img.permute(1, 2, 0))  # Convert CxHxW to HxWxC for display
    ax.set_title(f"Timestep {t}\nLabel: {label_name}")
    ax.axis('off')

plt.tight_layout()
plt.show()
