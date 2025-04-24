# Autoencoder Chatbot Emotion Mapping

## 🧠 Project Overview
This project trains an autoencoder to convert one facial emotion into another (e.g., smile → neutral) using a custom image dataset. It supports chatbot applications that involve visual emotion understanding.

## 📁 Dataset
- The dataset consists of 20+ pairs of images.
- Each pair contains an input image (smiling face) and a target output image (neutral face).
- Stored in:
  - `dataset/input/`
  - `dataset/output/`

## ⚙️ Model Architecture
CNN-based autoencoder:
- **Encoder**: 3 convolutional layers
- **Latent Space**: 64-dimensional bottleneck
- **Decoder**: 3 transposed convolutional layers

## 📉 Training
- Loss: MSELoss
- Optimizer: Adam
- Epochs: 100
- Training/Validation split: 80/20

## ✅ Results
### Example Results:
| Input (Smile) | Output (Predicted) | Target (Neutral) |
|---------------|--------------------|------------------|
| ![](input.jpg) | ![](output_pred.jpg) | ![](output_true.jpg) |

### Training Loss Curve:
![](loss_plot.png)

## 📦 How to Run
```bash
pip install -r requirements.txt
python train.py
python predict.py --input "dataset/input/sample.png"
