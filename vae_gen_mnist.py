import torch, random, os
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from mnist_dataset import MnistDataset
from mylib import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CondVAE(nn.Module):
    def __init__(self, z_dim=20):
        super().__init__()
        self.z_dim = z_dim
        # encoder: 784+10 → 400 → 20 (μ) / 20 (logσ²)
        self.fc_enc = nn.Sequential(
            nn.Linear(794, 400), nn.ReLU(),
            nn.Linear(400, z_dim*2)   # 前 z_dim 是 μ，后 z_dim 是 logσ²
        )
        # decoder: z+10 → 400 → 784
        self.fc_dec = nn.Sequential(
            nn.Linear(z_dim+10, 400), nn.ReLU(),
            nn.Linear(400, 784),      nn.Sigmoid()
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        if os.path.exists(filepath := "cond_vae_model.pth"):
            checkpoint = torch.load(filepath, map_location=device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")

    def encode(self, x, y):
        xy = torch.cat([x.view(x.size(0), -1), y], 1)
        h = self.fc_enc(xy)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        zy = torch.cat([z, y], 1)
        return self.fc_dec(zy)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar

    def loss_fn(self, x, x_hat, mu, logvar):
        BCE = F.binary_cross_entropy(x_hat, x.view(x.size(0), -1), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def gen_one(self,digit):
        self.eval()
        with torch.no_grad():
            z = torch.randn(1, self.z_dim).to(device)
            y = torch.eye(10)[digit:digit+1].to(device)
            img = self.decode(z, y).view(28,28)
            return 1-img.cpu()

    def show_result(self):
        plt.figure(figsize=(10,8))
        for i in range(8):
            for j in range(10):
                plt.subplot(8,10,i*10+j+1)
                img = self.gen_one(j)
                plt.imshow(img, cmap='gray')
                plt.title(f"{j}")
                plt.axis('off')
        plt.tight_layout()    
        plt.show()
    def show_digital(self,digital):
        plt.figure(figsize=(10,4))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            img = self.gen_one(digital)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.title(f"Digital: {digital}")
        plt.tight_layout()
        plt.show()
    def save_model(self, filepath="cond_vae_model.pth"):
        """Save the model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'z_dim': self.z_dim
        }, filepath)
        print(f"Model saved to {filepath}")
    def train_mode(self):
        epochs = 100
        dataset = MnistDataset()
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=True)
        for i in range(epochs):
            total_loss = 0
            for y, x ,label_tensor in train_loader:
                y = torch.eye(10)[y].to(device)     # Convert labels to one-hot
                x = x.view(x.size(0), -1).to(device) # Flatten images to (batch_size x 784)
                x_hat, mu, logvar = self(x, y)   # Forward pass with both image and label
                loss = self.loss_fn(x, x_hat, mu, logvar)
                self.opt.zero_grad(); 
                loss.backward(); 
                self.opt.step()
                total_loss += loss.item()
            print(f"Epoch {i+1}/{epochs}, Loss: {total_loss/len(train_loader.dataset):.4f}")
        self.save_model()
        self.show_result()
    



if __name__ == "__main__":
    vae=CondVAE()
    #vae.show_result()
    vae.train_mode()