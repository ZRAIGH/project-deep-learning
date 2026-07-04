import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

# Konfigurasi Path Global
DATASET_DIR = r"C:\Users\zora\OneDrive\Documents\project-deep-learning\Dataset\Training"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("--- [INFO] Memuat Dataset untuk Pengecekan ---")
train_dataset = image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb" 
)

val_dataset = image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb"
)

class_names = train_dataset.class_names
print(f"\n[SUKSES] Kelas emosi yang terdeteksi ({len(class_names)} Kelas): {class_names}")

# Menampilkan 9 sampel gambar dari dataset
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
print("\n[INFO] Menampilkan jendela pop-up gambar sampel dataset...")
plt.show()