import torch
from diffusion_utils import betas, alphas, alpha_hat, T

@torch.no_grad()
def sample(model, steps=T, device="cpu"):
    model.eval()
    x = torch.randn((1, 3, 32, 32), device=device)
    noisy_images = []

    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t], device=device)
        beta = betas[t]
        alpha = alphas[t]
        alpha_h = alpha_hat[t]
        noise_pred = model(x, t_tensor)
        z = torch.randn_like(x) if t > 0 else 0
        x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_h) * noise_pred) + torch.sqrt(beta) * z
        if t == steps - 1:
            noisy_images.append(x)

    return x, noisy_images
