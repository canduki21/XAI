
# MNIST Second-Order Explanations 

#Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


# Setup

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)
g = torch.Generator()
g.manual_seed(torch.seed()) 


# Softplus Model R^(28 * 28) linear with scalar output

class SoftplusNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SoftplusNet().to(device)


# Data

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, generator=g) #Random number



# Training

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training...")
model.train()
for epoch in range(3):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed")


# Find test image

model.eval()
x0, y0 = None, None

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        if pred.item() == y.item():
            x0 = x.clone()
            y0 = y.item()
            break

assert x0 is not None, "No correctly classified test image found!"
print(f"Selected digit: {y0}")

# Display the image 

img = x0.detach().cpu().squeeze()

plt.figure(figsize=(3,3))
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title(f"Correctly classified digit: {y0}")
plt.show()

# Save image
plt.figure(figsize=(3,3))
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.savefig("outputs/selected_image.png", dpi=150, bbox_inches="tight")
plt.close()

# -------------------------
# Prepare scalar output f(x)
# -------------------------
x0.requires_grad_(True)

def f(x):
    logits = model(x)
    return logits.max(dim=1).values.squeeze()

# -------------------------
# Compute full Hessian 
# -------------------------
print("Computing Hessian ...")

output = f(x0)
grad = torch.autograd.grad(output, x0, create_graph=True)[0]
grad_flat = grad.view(-1)

n = grad_flat.numel()
H = torch.zeros(n, n, device=device)

for i in range(n):
    second_grad = torch.autograd.grad(
        grad_flat[i], x0, retain_graph=True
    )[0]
    H[i] = second_grad.view(-1)

H = H.detach().cpu().numpy()


# Eigenvalues calculation 

eigenvalues = np.linalg.eigvalsh(H)

print("Eigenvalue statistics:")
print("Count:", eigenvalues.shape[0])
print("Min:", eigenvalues.min())
print("Max:", eigenvalues.max())
print("Mean:", eigenvalues.mean())


# Plot of eigenvalue spectrum

plt.figure(figsize=(6,4))
plt.semilogy(np.abs(eigenvalues) + 1e-12, ".", markersize=2)
plt.title("Hessian Eigenvalue Spectrum (Softplus)")
plt.xlabel("Index")
plt.ylabel("|Eigenvalue| (log scale)")
plt.tight_layout()
plt.show()

print("Done. Outputs saved in ./outputs/")
