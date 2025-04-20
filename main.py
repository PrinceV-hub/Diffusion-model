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

# Visualize a few random images from the dataset
fig, axs = plt.subplots(2, 5, figsize=(6, 6))  # 2x5 grid for images
for i in range(10):
    img, label = dataset[i]  # Get image and label from dataset
    img = (img + 1) / 2  # Rescale from [-1, 1] to [0, 1] for visualization
    axs[i // 5, i % 5].imshow(img.permute(1, 2, 0))  # Convert CxHxW to HxWxC for display
    axs[i // 5, i % 5].set_title(dataset.classes[label])  # Set title as the class label
    axs[i // 5, i % 5].axis('off')  # Hide axes

plt.tight_layout()
plt.show()

# STEP 3: Define U-Net Model
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x, t):
        return self.model(x)

# STEP 4: Noise Diffusion Functions
T = 1000  # Total timesteps
betas = torch.linspace(1e-4, 0.02, T)  # Beta from 0.0001 to 0.02
alphas = 1 - betas  # 1 - beta_t
alpha_hat = torch.cumprod(alphas, dim=0)  # Cumulative product of alphas

# Function to add noise at each timestep
def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)  # Ensures the shape matches x_0

    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])  # Apply alpha_hat for timestep t
    sqrt_one_minus = torch.sqrt(1 - alpha_hat[t])  # Apply noise scaling for timestep t

    # Make sure that we scale the image and noise properly (same shape)
    x_t = sqrt_alpha_hat[:, None, None, None] * x_0 + sqrt_one_minus[:, None, None, None] * noise  # Add noise to image
    return x_t, noise  # Return both noisy image and the noise added

# STEP 5: Training Loop (Noise Prediction)
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}!")

# Model and optimizer setup
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train for 3 epochs (demo purpose)
for epoch in range(3):  # Run for 3 epochs
    for x, _ in tqdm(loader):  # Assuming 'loader' is defined
        x = x.to(device)  # Move data to device (GPU/CPU)

        # Random timestep for each image in the batch
        t = torch.randint(0, T, (x.shape[0],), device=device).long()

        # Forward diffusion step: add noise to the image
        x_noisy, noise = forward_diffusion_sample(x, t)

        # Model prediction of the noise
        pred_noise = model(x_noisy, t)

        # Calculate loss (Mean Squared Error between predicted noise and actual noise)
        loss = F.mse_loss(pred_noise, noise)

        # Backpropagation and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

    # Print the loss after each epoch
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # Save checkpoint after each epoch
    save_checkpoint(epoch + 1, model, optimizer, loss.item(), checkpoint_path=f"checkpoint_epoch_{epoch + 1}.pth")

# STEP 6: Sampling from Noise (Reverse Process)
@torch.no_grad()
def sample(model, steps=T):
    model.eval()
    x = torch.randn((1, 3, 32, 32), device=device)  # Start from random noise
    noisy_images = []  # List to hold the noisy images during reverse process

    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t], device=device)
        beta = betas[t]
        alpha = alphas[t]
        alpha_h = alpha_hat[t]

        noise_pred = model(x, t_tensor)
        if t > 0:
            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        else:
            z = 0

        # Reverse process: denoising
        x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_h) * noise_pred) + torch.sqrt(beta) * z

        # Capture the noisy image at the first step (when t=999)
        if t == steps - 1:  # Capture the first noisy image (i.e., the one at the last timestep)
            noisy_images.append(x)

    return x, noisy_images

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
