import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.autoencoder import Autoencoder
import sys

model = Autoencoder()
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

img_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/input/sample.png'
img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output_tensor = model(input_tensor)

output_img = transforms.ToPILImage()(output_tensor.squeeze(0))


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img)
ax1.set_title('Input')
ax2.imshow(output_img)
ax2.set_title('Predicted Output')
plt.savefig('prediction_result.png')