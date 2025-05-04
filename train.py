import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from diffusion_utils import forward_diffusion_sample, T

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}!")

def train(model, optimizer, loader, device):
    for epoch in range(3):
        for x, _ in tqdm(loader):
            x = x.to(device)
            t = torch.randint(0, T, (x.shape[0],), device=device).long()
            x_noisy, noise = forward_diffusion_sample(x, t)
            pred_noise = model(x_noisy, t)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")
        save_checkpoint(epoch + 1, model, optimizer, loss.item(), f"checkpoint_epoch_{epoch + 1}.pth")
