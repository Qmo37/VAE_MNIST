"""
VAE MNIST - Variational Autoencoder for Beginners
==================================================

This is a beginner-friendly implementation of a Variational Autoencoder (VAE)
for reconstructing MNIST handwritten digits.

What is a VAE?
- VAE = Variational Autoencoder
- It's a type of neural network that learns to compress data (encode) and
  then recreate it (decode)
- Unlike regular autoencoders, VAEs learn a probabilistic representation
- This allows them to generate new, similar data

Key Components:
1. Encoder: Compresses 28x28 images (784 pixels) into a small latent space (20 dimensions)
2. Decoder: Reconstructs images from the latent space back to 784 pixels
3. Reparameterization Trick: Allows backpropagation through random sampling

Architecture Overview:
Input Image (28x28=784) → Encoder → Latent Space (20D) → Decoder → Reconstructed Image (784)
"""

# ===========================
# 1. IMPORT LIBRARIES
# ===========================

# Import all the libraries we need
import torch              # Main PyTorch library for deep learning
import torch.nn as nn     # Neural network building blocks (layers, etc.)
import torch.nn.functional as F  # Activation functions (ReLU, sigmoid, etc.)
import torch.optim as optim      # Optimizers (Adam, SGD, etc.)
from torch.utils.data import DataLoader  # For loading data in batches
from torchvision import datasets, transforms  # MNIST dataset and data preprocessing
import matplotlib.pyplot as plt  # For plotting graphs and images
import numpy as np        # For numerical operations
import os

# Fix a common warning that appears on some systems
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ===========================
# 2. SETUP DEVICE AND SEEDS
# ===========================

# Choose device: GPU if available (faster), otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Why set random seeds?
# - Makes results reproducible (same results every time you run the code)
# - Important for debugging and comparing different approaches
torch.manual_seed(42)  # Set seed for PyTorch random operations
np.random.seed(42)     # Set seed for NumPy random operations

# ===========================
# 3. VAE MODEL DEFINITION
# ===========================

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) Class

    This class defines our VAE model with:
    - Encoder: Takes images and produces mean (mu) and log-variance (logvar)
    - Decoder: Takes latent codes and reconstructs images
    - Reparameterization: Samples from the latent distribution

    Parameters:
    - input_dim: Size of input (784 for 28x28 MNIST images)
    - hidden_dim: Size of hidden layers (400 is a good default)
    - latent_dim: Size of latent space (20 dimensions for compression)
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        # Call parent class constructor (required for PyTorch modules)
        super(VAE, self).__init__()

        # ===== ENCODER LAYERS =====
        # The encoder compresses images into a latent representation

        # First layer: 784 (flattened image) → 400 (hidden layer)
        # This layer learns features from the raw pixel data
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Output layers: 400 → 20 (latent dimensions)
        # We need TWO outputs: mean (mu) and log-variance (logvar)
        # This is because VAE learns a DISTRIBUTION, not just a single point
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # Mean of latent distribution
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log-variance of latent distribution

        # ===== DECODER LAYERS =====
        # The decoder reconstructs images from latent codes

        # First layer: 20 (latent) → 400 (hidden layer)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)

        # Output layer: 400 → 784 (reconstructed image)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        ENCODER: Convert input images to latent distribution parameters

        Input: x (batch_size, 784) - flattened images
        Output: mu, logvar (batch_size, 20) - distribution parameters

        Why mu and logvar?
        - mu: Mean of the latent distribution
        - logvar: Log of variance (more stable than raw variance)
        """
        # Apply ReLU activation to first layer
        # ReLU(x) = max(0, x) - removes negative values, helps with learning
        h1 = F.relu(self.fc1(x))

        # Get distribution parameters (no activation needed for final layer)
        mu = self.fc_mu(h1)      # Mean can be any real number
        logvar = self.fc_logvar(h1)  # Log-variance can be any real number

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        REPARAMETERIZATION TRICK: Sample from latent distribution

        This is the KEY innovation of VAEs!

        Problem: We can't backpropagate through random sampling
        Solution: z = mu + std * epsilon, where epsilon ~ N(0,1)

        This lets us:
        1. Sample from the distribution: z ~ N(mu, std²)
        2. Still compute gradients through mu and std

        Input: mu, logvar (batch_size, 20)
        Output: z (batch_size, 20) - sampled latent codes
        """
        # Convert log-variance to standard deviation
        # std = sqrt(variance) = sqrt(exp(logvar)) = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)

        # Sample random noise: epsilon ~ N(0, 1)
        # torch.randn_like(std) creates random numbers with same shape as std
        eps = torch.randn_like(std)

        # Reparameterization: z = mu + std * epsilon
        # This samples from N(mu, std²) but allows gradients through mu and std
        return mu + eps * std

    def decode(self, z):
        """
        DECODER: Convert latent codes back to images

        Input: z (batch_size, 20) - latent codes
        Output: reconstructed images (batch_size, 784)
        """
        # Apply ReLU to hidden layer
        h3 = F.relu(self.fc3(z))

        # Apply sigmoid to output layer
        # Sigmoid squashes values to [0,1] range, perfect for image pixels
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """
        FORWARD PASS: Complete VAE pipeline

        This method is called automatically when you use model(data)

        Steps:
        1. Encode input to get distribution parameters
        2. Sample from the distribution (reparameterization)
        3. Decode the sample to reconstruct the input

        Input: x (batch_size, 1, 28, 28) - original images
        Output: recon_x, mu, logvar - reconstructed images and distribution params
        """
        # Flatten images: (batch_size, 1, 28, 28) → (batch_size, 784)
        x_flat = x.view(-1, 784)

        # Encode: get distribution parameters
        mu, logvar = self.encode(x_flat)

        # Sample: get latent codes using reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode: reconstruct images from latent codes
        recon_x = self.decode(z)

        # Return everything (we need mu and logvar for loss calculation)
        return recon_x, mu, logvar

# ===========================
# 4. LOSS FUNCTION
# ===========================

def vae_loss(recon_x, x, mu, logvar):
    """
    VAE LOSS FUNCTION: Reconstruction + Regularization

    VAE loss has two parts:
    1. Reconstruction Loss: How well can we reconstruct the input?
    2. KL Divergence: How close is our latent distribution to N(0,1)?

    Why two parts?
    - Reconstruction loss: Ensures we don't lose information
    - KL divergence: Prevents overfitting and enables generation

    Parameters:
    - recon_x: Reconstructed images (batch_size, 784)
    - x: Original images (batch_size, 1, 28, 28)
    - mu: Mean of latent distribution (batch_size, 20)
    - logvar: Log-variance of latent distribution (batch_size, 20)
    """

    # ===== RECONSTRUCTION LOSS =====
    # Measures how different the reconstructed image is from the original
    # Uses Binary Cross Entropy (BCE) since pixels are in [0,1] range

    # Flatten original images to match reconstructed shape
    x_flat = x.view(-1, 784)

    # Calculate BCE loss (sum over all pixels and all images in batch)
    BCE = F.binary_cross_entropy(recon_x, x_flat, reduction='sum')

    # ===== KL DIVERGENCE LOSS =====
    # Measures how different our learned distribution is from N(0,1)
    # This acts as a regularizer - prevents the model from "cheating"

    # Mathematical formula for KL divergence between N(mu, sigma²) and N(0,1):
    # KL = 0.5 * sum(1 + log(sigma²) - mu² - sigma²)
    # Since we have log(sigma²) = logvar, this becomes:
    # KL = 0.5 * sum(1 + logvar - mu² - exp(logvar))

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # ===== TOTAL LOSS =====
    # Combine both losses
    # Note: Some implementations use a weight (beta) for KLD, but we use 1.0
    return BCE + KLD

# ===========================
# 5. DATA LOADING
# ===========================

print("Loading MNIST dataset...")
print("MNIST contains:")
print("- 60,000 training images of handwritten digits (0-9)")
print("- 10,000 test images")
print("- Each image is 28x28 pixels (grayscale)")

# Data preprocessing
# transforms.ToTensor() converts PIL images to PyTorch tensors and scales to [0,1]
transform = transforms.Compose([
    transforms.ToTensor()  # Convert to tensor and normalize to [0,1]
])

# Download and load MNIST dataset
# train=True gets training data, train=False gets test data
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Create data loaders
# DataLoader batches the data and shuffles it for training
batch_size = 128  # Process 128 images at a time

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}")
print(f"Training batches: {len(train_loader)}")

# ===========================
# 6. MODEL INITIALIZATION
# ===========================

# Create the VAE model and move it to our device (GPU or CPU)
model = VAE().to(device)

# Count the number of parameters in our model
total_params = sum(p.numel() for p in model.parameters())
print(f'\nModel created with {total_params:,} parameters')

# Let's break down the architecture:
print("\nModel Architecture:")
print("Encoder: 784 → 400 → (20 + 20)  [input → hidden → (mu, logvar)]")
print("Decoder: 20 → 400 → 784         [latent → hidden → output]")

# ===== OPTIMIZER =====
# Adam is an advanced version of gradient descent
# It adapts the learning rate for each parameter automatically
# lr=0.001 is a good default learning rate for Adam

optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"Using Adam optimizer with learning rate: 0.001")

# ===========================
# 7. TRAINING FUNCTION
# ===========================

def train(epoch):
    """
    Train the model for one epoch

    An epoch = one complete pass through all training data

    Steps for each batch:
    1. Forward pass: get reconstructions and distribution parameters
    2. Calculate loss: reconstruction + KL divergence
    3. Backward pass: compute gradients
    4. Update parameters: apply gradients with optimizer

    Returns: average loss for this epoch
    """
    # Set model to training mode
    # This enables dropout and batch normalization (not used in this simple VAE)
    model.train()

    total_loss = 0  # Keep track of total loss across all batches

    # Loop through all batches in the training set
    for batch_idx, (data, _) in enumerate(train_loader):
        # Note: we ignore labels (_) since VAE is unsupervised

        # Move data to device (GPU or CPU)
        data = data.to(device)

        # Clear gradients from previous batch
        # PyTorch accumulates gradients, so we must clear them each time
        optimizer.zero_grad()

        # ===== FORWARD PASS =====
        # Pass data through the model
        recon_batch, mu, logvar = model(data)

        # ===== CALCULATE LOSS =====
        loss = vae_loss(recon_batch, data, mu, logvar)

        # ===== BACKWARD PASS =====
        # Calculate gradients using backpropagation
        loss.backward()

        # ===== UPDATE PARAMETERS =====
        # Apply gradients to update model parameters
        optimizer.step()

        # Add this batch's loss to total (for calculating average)
        total_loss += loss.item()

        # Print progress every 200 batches
        if batch_idx % 200 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item() / len(data):.6f}')

    # Calculate average loss per sample
    avg_loss = total_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')

    return avg_loss

# ===========================
# 8. TRAINING LOOP
# ===========================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

print("\nWhat happens during training:")
print("1. Show images to the encoder")
print("2. Encoder learns to compress images to 20 numbers")
print("3. Decoder learns to reconstruct images from those 20 numbers")
print("4. Gradually get better at reconstruction")

epochs = 5  # Number of complete passes through the dataset
losses = []  # Keep track of loss for each epoch

for epoch in range(1, epochs + 1):
    print(f"\n--- EPOCH {epoch}/{epochs} ---")
    avg_loss = train(epoch)
    losses.append(avg_loss)

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)

# ===========================
# 9. RESULTS VISUALIZATION
# ===========================

print("\nGenerating reconstruction results...")

# Set model to evaluation mode (disables dropout, etc.)
model.eval()

# Disable gradient computation for faster inference
with torch.no_grad():
    # Get a batch of test data
    data, _ = next(iter(test_loader))
    data = data.to(device)

    # Select first 8 samples for visualization
    test_samples = data[:8]

    # Generate reconstructions
    recon_samples, _, _ = model(test_samples)

    # ===== CREATE COMPARISON PLOT =====
    fig, axes = plt.subplots(2, 8, figsize=(15, 4))

    for i in range(8):
        # Top row: Original images
        axes[0, i].imshow(test_samples[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')

        # Bottom row: Reconstructed images
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
# 10. SUMMARY AND EXPLANATION
# ===========================

print(f"\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

print(f"Dataset: MNIST (28x28 handwritten digit images)")
print(f"Model: Variational Autoencoder")
print(f"Architecture: 784 → 400 → 20 → 400 → 784")
print(f"Optimizer: Adam (learning rate = 0.001)")
print(f"Training epochs: {epochs}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss improvement: {losses[0] - losses[-1]:.4f}")

print(f"\n" + "="*60)
print("WHAT DID WE ACCOMPLISH?")
print("="*60)

print("✓ Encoder: Learned to compress 784-pixel images into just 20 numbers")
print("✓ Decoder: Learned to reconstruct images from those 20 numbers")
print("✓ Reparameterization: Can sample new points in latent space")
print("✓ Generation: Can create new digit-like images (not shown in this demo)")

print(f"\n" + "="*60)
print("KEY CONCEPTS YOU LEARNED")
print("="*60)

print("1. COMPRESSION: 784 pixels → 20 latent variables (39x compression!)")
print("2. RECONSTRUCTION: VAE learns to recreate original images")
print("3. PROBABILISTIC: VAE learns distributions, not just single points")
print("4. GENERATION: Can create new images by sampling latent space")
print("5. REGULARIZATION: KL divergence keeps latent space well-behaved")

print(f"\n" + "="*60)
print("NEXT STEPS FOR EXPLORATION")
print("="*60)

print("• Try different latent dimensions (10, 50, 100)")
print("• Experiment with different architectures (more layers, different sizes)")
print("• Try β-VAE (weight the KL divergence term)")
print("• Generate new images by sampling from latent space")
print("• Interpolate between two images in latent space")
print("• Try on different datasets (CIFAR-10, CelebA)")

print(f"\nVAE implementation completed! 🎉")
