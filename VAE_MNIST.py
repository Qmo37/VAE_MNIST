"""
VAE MNIST - Assignment Implementation
Simple and clean implementation following exact assignment requirements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Fix OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ===========================
# VAE Model (Assignment Requirements)
# ===========================
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # Encoder: 784 -> 400 -> (mu, logvar)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent_dim -> 400 -> 784
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Encoder: converts input to latent space parameters"""
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decoder: converts latent variable back to image"""
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ===========================
# Loss Function (Standard VAE Loss)
# ===========================
def vae_loss(recon_x, x, mu, logvar):
    """
    VAE loss = Reconstruction loss + KL divergence
    """
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# ===========================
# Data Loading (MNIST)
# ===========================
print("Loading MNIST dataset...")

# Data transforms
transform = transforms.Compose([transforms.ToTensor()])

# Load datasets
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ===========================
# Model Initialization
# ===========================
model = VAE().to(device)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

# Adam optimizer (as required)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===========================
# Training Function
# ===========================
def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = model(data)

        # Calculate loss
        loss = vae_loss(recon_batch, data, mu, logvar)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Progress logging
        if batch_idx % 200 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    # Calculate average loss for epoch
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

# ===========================
# Training Loop
# ===========================
print("\nStarting training...")
print("="*50)

epochs = 5  # As mentioned in original requirements
losses = []

for epoch in range(1, epochs + 1):
    avg_loss = train(epoch)
    losses.append(avg_loss)

print("="*50)
print("Training completed!")

# ===========================
# Results Visualization
# ===========================
print("\nGenerating reconstruction results...")

model.eval()
with torch.no_grad():
    # Get test data
    data, _ = next(iter(test_loader))
    data = data.to(device)

    # Select 8 samples for visualization
    test_samples = data[:8]

    # Reconstruct
    recon_samples, _, _ = model(test_samples)

    # Create visualization
    fig, axes = plt.subplots(2, 8, figsize=(15, 4))

    for i in range(8):
        # Original images
        axes[0, i].imshow(test_samples[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')

        # Reconstructed images
        axes[1, i].imshow(recon_samples[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')

    plt.suptitle('VAE MNIST Reconstruction Results', fontsize=16)
    plt.tight_layout()

    # Save the results
    plt.savefig('reconstruction.png', dpi=100, bbox_inches='tight')
    print("Reconstruction results saved as 'reconstruction.png'")
    plt.show()

# ===========================
# Summary
# ===========================
print(f"\nTraining Summary:")
print(f"• Dataset: MNIST (28x28 images)")
print(f"• Model: Simple VAE (784→400→20→400→784)")
print(f"• Optimizer: Adam (lr=0.001)")
print(f"• Epochs: {epochs}")
print(f"• Final loss: {losses[-1]:.4f}")
print(f"• Loss improvement: {losses[0] - losses[-1]:.4f}")
print(f"• Output: reconstruction.png")

print(f"\nAssignment requirements fulfilled:")
print(f"  ✓ Used MNIST dataset")
print(f"  ✓ Implemented Encoder (784D → mu, logvar)")
print(f"  ✓ Used reparameterization trick")
print(f"  ✓ Implemented Decoder (z → 784D)")
print(f"  ✓ Used Adam optimizer")
print(f"  ✓ Displayed epoch losses")
print(f"  ✓ Generated reconstruction visualization")

print(f"\nSimple VAE implementation completed!")
