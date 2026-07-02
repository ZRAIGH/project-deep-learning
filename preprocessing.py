import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. Path dataset kamu
dataset_dir = r"C:\Users\USER\OneDrive\Desktop\Kuliah\Semester 6\Deep Learning\Proyek\Dataset\Training"

# Parameter Model
BATCH_SIZE = 32
IMG_SIZE = (224, 224) # Ukuran input standar untuk Transfer Learning

print("--- Memuat Dataset Training (80%) ---")
train_dataset = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb" 
)

print("\n--- Memuat Dataset Validasi (20%) ---")
val_dataset = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

# Mengambil nama folder sebagai label/kelas emosi
class_names = train_dataset.class_names
print(f"\n[INFO] Kelas emosi yang terdeteksi: {class_names}")

# 2. Normalisasi Data (Skala pixel 0-255 menjadi 0.0-1.0)
normalization_layer = layers.Rescaling(1./255)
normalized_train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
normalized_val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

print("[INFO] Preprocessing & Normalisasi berhasil dikonfigurasi.")

# 3. Menampilkan 9 contoh gambar dari dataset untuk memastikan data aman
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
print("\n[INFO] Menampilkan pop-up gambar sampel dataset...")
plt.show()