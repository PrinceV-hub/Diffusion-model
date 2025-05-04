# 🌀 DDPM on CIFAR-10

This repository implements a simple **Denoising Diffusion Probabilistic Model (DDPM)** using PyTorch on the CIFAR-10 dataset. It uses a simplified U-Net-like CNN to learn how to denoise images corrupted with Gaussian noise over time.

---

## 📂 Project Structure
ddpm_cifar10_diffusion/

├── install_imports.py # Setup & required imports (for Colab)

├── load_data.py # Load CIFAR-10 dataset and visualize samples

├── model.py # Simple CNN (U-Net-like) to predict noise

├── diffusion_utils.py # Forward diffusion noise functions

├── train.py # Noise prediction training loop

├── reverse_sample.py # Reverse denoising process from noise

├── sample_plot.py # Visualize how noise corrupts an image

├── main.py # Trains model using all components

├── requirements.txt # Dependency list

---

## 📦 Dependencies

Install them using:

```bash
pip install -r requirements.txt
```
## Run the code
```bash
python main.py
```
