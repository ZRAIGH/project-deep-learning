import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

dataset_dir = r"C:\Users\USER\OneDrive\Desktop\Kuliah\Semester 6\Deep Learning\Proyek\Dataset\Training"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# WAJIB tambahkan shuffle=True di kedua dataset agar pembagian 80/20 adil mendapat semua emosi
train_dataset = image_dataset_from_directory(
    dataset_dir, validation_split=0.2, subset="training", seed=123,
    shuffle=True, # <--- Pastikan ini True
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb"
)

val_dataset = image_dataset_from_directory(
    dataset_dir, validation_split=0.2, subset="validation", seed=123,
    shuffle=True, # <--- Ubah ini menjadi True juga!
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb"
)

# Ambil jumlah kelas emosi (harus ada 7)
num_classes = len(train_dataset.class_names)

# Normalisasi terintegrasi di dalam tf.data untuk efisiensi memori
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)


# --- 2. MEMBANGUN MODEL (TRANSFER LEARNING MOBILENETV2) ---
print("\n[INFO] Mengunduh & mengonfigurasi MobileNetV2...")
# Load base model MobileNetV2 tanpa layer klasifikasi atasnya (include_top=False)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model agar bobot prateks tidak berubah dulu saat latihan awal
base_model.trainable = False

# Membuat arsitektur model baru kita
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), # Mencegah overfitting
    layers.Dense(num_classes, activation='softmax') # Output layer untuk 7 kelas emosi
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# --- 3. PROSES PELATIHAN (TRAINING) ---
# Kita coba 5 Epochs dulu sebagai uji coba awal apakah modelnya belajar
EPOCHS = 5
print(f"\n[INFO] Memulai training untuk {EPOCHS} epoch...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)


# --- 4. MENYIMPAN MODEL TERBAIK ---
# Sesuai deliverable tugas wajib menyimpan model (.h5)
model_save_path = "model_emotion_mobilenet.h5"
model.save(model_save_path)
print(f"\n[INFO] Model sukses dilatih dan disimpan di: {model_save_path}")


# --- 5. VISUALISASI HASIL TRAINING ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Akurasi Model')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss Model')
plt.show()