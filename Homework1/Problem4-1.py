# =========================================================
# DOGS vs CATS — Raw Gradients, SmoothGrad, VarGrad
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = '/Users/canduki21/Fall-2024-Deep-Learning/data/dogs-vs-cats/train/train'


BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.001

NUM_TEST_IMAGES = 5
N_SAMPLES = 20
NOISE_LEVEL = 0.15

device = "cpu"

# =========================================================
# DATASET
# =========================================================

class DogsVsCatsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir)
                       if f.endswith(('.jpg', '.jpeg', '.png'))
                       and f.startswith(('dog', 'cat'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = 0 if img_name.startswith("dog") else 1

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

full_dataset = DogsVsCatsDataset(DATA_DIR, transform=train_transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset,
                                           [train_size, test_size])

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

# =========================================================
# MODEL
# =========================================================

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# =========================================================
# TRAIN
# =========================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()
for epoch in range(NUM_EPOCHS):
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    print(f"Epoch {epoch+1} Accuracy: {100*correct/total:.2f}%")

# =========================================================
# EXPLANATION METHODS
# Scalar = predicted class logit
# Postprocessing: absolute value + channel mean + normalization
# =========================================================

def compute_gradient(model, image, target_class):
    image = image.clone().detach().requires_grad_(True)
    output = model(image)
    score = output[0, target_class]

    model.zero_grad()
    score.backward()

    return image.grad[0].cpu()


def smoothgrad(model, image, target_class):
    grads = []

    for _ in range(N_SAMPLES):
        noise = torch.randn_like(image) * NOISE_LEVEL
        noisy_image = (image + noise).clone().detach().requires_grad_(True)

        output = model(noisy_image)
        score = output[0, target_class]

        model.zero_grad()
        score.backward()

        grads.append(noisy_image.grad[0].cpu())

    return torch.mean(torch.stack(grads), dim=0)


def vargrad(model, image, target_class):
    grads = []

    for _ in range(N_SAMPLES):
        noise = torch.randn_like(image) * NOISE_LEVEL
        noisy_image = (image + noise).clone().detach().requires_grad_(True)

        output = model(noisy_image)
        score = output[0, target_class]

        model.zero_grad()
        score.backward()

        grads.append(noisy_image.grad[0].cpu())

    grads = torch.stack(grads)
    return torch.var(grads, dim=0)


# =========================================================
# VISUALIZATION
# =========================================================

def process_grad(grad):
    grad = torch.abs(grad)
    grad = grad.mean(dim=0)
    grad = (grad - grad.min())/(grad.max()-grad.min()+1e-8)
    return grad.numpy()

def denormalize(img):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img = img.cpu()*std + mean
    return torch.clamp(img,0,1).permute(1,2,0).numpy()


model.eval()

# =========================================================
# GENERATE EXPLANATIONS FOR 5 IMAGES
# =========================================================

count = 0
for image, label in test_dataset:
    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_input)
        pred = output.argmax(dim=1).item()
        conf = torch.softmax(output, dim=1)[0,pred].item()

    raw = compute_gradient(model, image_input, pred)
    smooth = smoothgrad(model, image_input, pred)
    var = vargrad(model, image_input, pred)

    fig, axes = plt.subplots(1,4,figsize=(16,4))

    axes[0].imshow(denormalize(image))
    axes[0].set_title(f"Pred: {['Dog','Cat'][pred]} ({conf:.2%})")
    axes[0].axis("off")

    for ax, grad, title in zip(
        axes[1:],
        [raw, smooth, var],
        ["Raw", "SmoothGrad", "VarGrad"]
    ):
        ax.imshow(process_grad(grad), cmap="hot")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    count += 1
    if count == NUM_TEST_IMAGES:
        break


