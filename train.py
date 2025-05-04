import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from diffusion_utils import forward_diffusion_sample, T
import matplotlib.pyplot as plt
import os

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}!")

def save_training_image(x, epoch, batch_idx):
    # Create output directory if it doesn't exist
    os.makedirs('training_outputs', exist_ok=True)
    
    # Convert tensor to numpy and rescale
    img = (x[0].cpu().detach().numpy() + 1) / 2
    img = img.transpose(1, 2, 0)
    
    # Save image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'training_outputs/epoch_{epoch}_batch_{batch_idx}.png')
    plt.close()

def train(model, optimizer, loader, device):
    for epoch in range(3):
        for batch_idx, (x, _) in enumerate(tqdm(loader)):
            x = x.to(device)
            t = torch.randint(0, T, (x.shape[0],), device=device).long()
            x_noisy, noise = forward_diffusion_sample(x, t)
            pred_noise = model(x_noisy, t)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save sample image every 100 batches
            if batch_idx % 100 == 0:
                save_training_image(x, epoch, batch_idx)

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")
        save_checkpoint(epoch + 1, model, optimizer, loss.item(), f"checkpoint_epoch_{epoch + 1}.pth")
