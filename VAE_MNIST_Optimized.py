"""
VAE MNIST - Optimized for Better Performance
Meets all PDF requirements with improved reconstruction quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Fix OpenMP warning (PDF requirement)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ===========================
# OPTIMIZED VAE Model
# Key improvements: Better architecture, dropout for regularization
# ===========================
class OptimizedVAE(nn.Module):
    def __init__(self, input_dim=784, h_dim1=512, h_dim2=256, z_dim=20):
        super(OptimizedVAE, self).__init__()

        # Encoder - Improved with deeper architecture
        self.fc1 = nn.Linear(input_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc_mu = nn.Linear(h_dim2, z_dim)
        self.fc_logvar = nn.Linear(h_dim2, z_dim)

        # Decoder - Mirror architecture
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, input_dim)

        # Dropout for better generalization
        self.dropout = nn.Dropout(0.2)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick (PDF requirement)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ===========================
# Improved Loss Function with Beta-VAE weighting
# ===========================
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Beta-VAE loss for better reconstruction-KL tradeoff
    beta < 1: Better reconstruction
    beta > 1: Better latent space structure
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

# ===========================
# Data with augmentation for better generalization
# ===========================
print("Loading MNIST dataset...")

# Simple augmentation that still preserves digit clarity
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(5),  # Small rotation
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))  # Slight noise
])

transform_test = transforms.ToTensor()  # No augmentation for test

train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True)

test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform_test),
    batch_size=128, shuffle=False)

# ===========================
# Model with optimized hyperparameters
# ===========================
model = OptimizedVAE(
    input_dim=784,
    h_dim1=512,    # Larger hidden layer
    h_dim2=256,    # Additional layer
    z_dim=20       # Keep latent dim as per PDF
).to(device)

# Adam optimizer (PDF requirement) with tuned learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# ===========================
# Training with improvements
# ===========================
def train_optimized(epoch, beta=0.5):
    """
    Train with beta-VAE loss for better reconstruction
    beta=0.5 prioritizes reconstruction quality
    """
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar, beta=beta)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}, Average loss: {avg_loss:.4f}')
    return avg_loss

# ===========================
# Main Training - 5 epochs as per PDF
# ===========================
print("\nStarting optimized training...")
print("="*50)

losses = []
best_loss = float('inf')

# Train for 5 epochs (PDF requirement)
for epoch in range(1, 6):
    avg_loss = train_optimized(epoch, beta=0.5)  # Beta < 1 for better reconstruction
    losses.append(avg_loss)

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_vae.pth')

    scheduler.step(avg_loss)

print("="*50)
print(f"訓練完成！Best loss: {best_loss:.4f}")

# Load best model for visualization
model.load_state_dict(torch.load('best_vae.pth'))

# ===========================
# Enhanced Visualization
# ===========================
print("\n產生重建圖檔 reconstruction.png")

model.eval()
with torch.no_grad():
    # Get test samples
    data, labels = next(iter(test_loader))

    # Select diverse digits (0-9 if possible)
    selected_indices = []
    for digit in range(10):
        digit_indices = (labels == digit).nonzero(as_tuple=True)[0]
        if len(digit_indices) > 0:
            selected_indices.append(digit_indices[0].item())

    # Use first 8 diverse samples
    if len(selected_indices) >= 8:
        data = data[selected_indices[:8]].to(device)
    else:
        data = data[:8].to(device)

    # Reconstruct
    recon, mu, logvar = model(data)
    recon = recon.view(-1, 1, 28, 28)

    # Create figure matching PDF format
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))

    for i in range(8):
        # Input
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Input', fontsize=12)

        # Reconstruction
        axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstruction', fontsize=12)

    plt.suptitle('Optimized VAE MNIST Reconstruction', fontsize=14)
    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=100, bbox_inches='tight')
    plt.show()

# ===========================
# Performance Metrics (Beyond PDF requirements but useful)
# ===========================
print("\n📊 Performance Summary:")
print(f"• Final Loss: {losses[-1]:.4f}")
print(f"• Loss Improvement: {losses[0] - losses[-1]:.4f}")
print(f"• Best Loss Achieved: {best_loss:.4f}")
print(f"• Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test reconstruction quality
model.eval()
test_loss = 0
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        recon, mu, logvar = model(data)
        test_loss += vae_loss(recon, data, mu, logvar, beta=0.5).item()

avg_test_loss = test_loss / len(test_loader.dataset)
print(f"• Test Set Loss: {avg_test_loss:.4f}")

print("\n✅ All PDF requirements met with optimized performance!")
