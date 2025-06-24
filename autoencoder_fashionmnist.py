import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from fashion_dataset import FashionMNISTFromFile  # Kendi veri yükleyicin
import numpy as np

latent_dim = 64
epochs = 50
batch_size = 128
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


train_dataset = FashionMNISTFromFile('fashion-mnist/data/fashion', kind='train')
test_dataset = FashionMNISTFromFile('fashion-mnist/data/fashion', kind='t10k')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = Autoencoder(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device).float()

        noisy_images = images + 0.2 * torch.randn_like(images)
        noisy_images = torch.clamp(noisy_images, 0., 1.)

        outputs = model(noisy_images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    scheduler.step()
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")


model.eval()
with torch.no_grad():
    sample_images, _ = next(iter(test_loader))
    sample_images = sample_images.to(device).float()
    recon_images = model(sample_images)
    test_loss = criterion(recon_images, sample_images).item()
    print(f"\nTest Reconstruction Loss: {test_loss:.4f}")


def show_reconstruction(original, reconstructed, num=10):
    plt.figure(figsize=(20, 4))
    for i in range(num):
        ax = plt.subplot(2, num, i + 1)
        plt.imshow(original[i].cpu().squeeze(), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')

        ax = plt.subplot(2, num, i + 1 + num)
        plt.imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        ax.set_title("Reconstructed")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("reconstruction_result.png")
    plt.show()

show_reconstruction(sample_images, recon_images)


with torch.no_grad():
    z = torch.randn(10, latent_dim).to(device)
    generated_images = model.decoder(z)


def show_generated(images):
    plt.figure(figsize=(15, 2))
    for i in range(images.size(0)):
        ax = plt.subplot(1, images.size(0), i + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        ax.set_title("Sample")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("generated_samples.png")
    plt.show()

show_generated(generated_images)
