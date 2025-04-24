# Autoencoder Chatbot Pemetaaan Emosi

## 🧠 Deskripsi Proyek
Proyek ini menggunakan autoencoder berbasis Convolutional Neural Network (CNN) untuk mentransformasi ekspresi wajah dari satu emosi ke emosi lainnya — misalnya dari ekspresi tersenyum menjadi ekspresi netral. Model ini dirancang untuk mendukung chatbot yang mampu mengenali dan memahami emosi wajah secara visual.

## 📁 Dataset
- Dataset terdiri dari lebih dari 20 pasang gambar.
- Setiap pasangan gambar terdiri dari:
  - **Input**: wajah dengan ekspresi tersenyum (`dataset/input/`)
  - **Output (Target)**: wajah dengan ekspresi netral (`dataset/output/`)
- Format gambar: PNG atau JPG

## ⚙️ Arsitektur Model
Model autoencoder dibangun dengan struktur sebagai berikut:

- **Encoder**
  - 3 Lapisan Konvolusi (Conv2D) + ReLU + BatchNorm
  - Bottleneck berdimensi 64
- **Decoder**
  - 3 Lapisan Transposed Convolution (ConvTranspose2D) + ReLU
  - Output layer menggunakan aktivasi sigmoid

## 📉 Pelatihan Model
- **Loss Function**: MSELoss (Mean Squared Error)
- **Optimizer**: Adam
- **Epochs**: 100
- **Batch Size**: 8
- **Pembagian Data**: 80% untuk pelatihan, 20% untuk validasi

Selama pelatihan, grafik loss akan disimpan dalam file `loss_plot.png`.


## 🚀 Cara Menjalankan
### 1. Install Dependensi
```bash
pip install -r requirements.txt
