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
