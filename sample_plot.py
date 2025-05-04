import matplotlib.pyplot as plt
import torch
import random
from diffusion_utils import forward_diffusion_sample, T

def plot_forward_timesteps(dataset):
    random_index = random.randint(0, len(dataset) - 1)
    sample_img, label = dataset[random_index]
    label_name = dataset.classes[label]

    timesteps = [0, 200, 400, 600, 800, T-1]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    original_img = (sample_img + 1) / 2
    axs[0, 0].imshow(original_img.permute(1, 2, 0))
    axs[0, 0].set_title(f"Original Image\\nLabel: {label_name}")
    axs[0, 0].axis('off')

    for i, t in enumerate(timesteps):
        x_t, _ = forward_diffusion_sample(sample_img.unsqueeze(0), torch.tensor([t]))
        noisy_img = (x_t.squeeze() + 1) / 2
        noisy_img = torch.clamp(noisy_img, 0, 1)
        ax = axs[i // 3, i % 3]
        ax.imshow(noisy_img.permute(1, 2, 0))
        ax.set_title(f"Timestep {t}\\nLabel: {label_name}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
