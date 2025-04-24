import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.autoencoder import Autoencoder

class CustomImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.output_paths = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_img = Image.open(self.input_paths[idx]).convert('RGB')
        output_img = Image.open(self.output_paths[idx]).convert('RGB')
        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)
        return input_img, output_img

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = CustomImageDataset('dataset/input', 'dataset/output', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
losses = []
for epoch in range(epochs):
    total_loss = 0
    for input_img, target_img in dataloader:
        output = model(input_img)
        loss = criterion(output, target_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
    losses.append(total_loss)

torch.save(model.state_dict(), 'autoencoder.pth')

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_plot.png")