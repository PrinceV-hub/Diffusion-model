T = 1000  # Total timesteps
betas = torch.linspace(1e-4, 0.02, T)  # Beta from 0.0001 to 0.02
alphas = 1 - betas  # 1 - beta_t
alpha_hat = torch.cumprod(alphas, dim=0)  # Cumulative product of alphas

# Function to add noise at each timestep
def forward_diffusion_sample(x_0, t):
    # Generate random noise with the same shape as x_0
    noise = torch.randn_like(x_0)  # Ensures the shape matches x_0

    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])  # Apply alpha_hat for timestep t
    sqrt_one_minus = torch.sqrt(1 - alpha_hat[t])  # Apply noise scaling for timestep t

    # Make sure that we scale the image and noise properly (same shape)
    x_t = sqrt_alpha_hat[:, None, None, None] * x_0 + sqrt_one_minus[:, None, None, None] * noise  # Add noise to image
    return x_t, noise  # Return both noisy image and the noise added
