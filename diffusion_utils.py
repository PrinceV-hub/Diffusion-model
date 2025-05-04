import torch

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_hat = torch.cumprod(alphas, dim=0)

def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])
    sqrt_one_minus = torch.sqrt(1 - alpha_hat[t])
    x_t = sqrt_alpha_hat[:, None, None, None] * x_0 + sqrt_one_minus[:, None, None, None] * noise
    return x_t, noise
