import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. Path Dataset dan Parameter
dataset_dir = r"C:\Users\USER\OneDrive\Desktop\Kuliah\Semester 6\Deep Learning\Proyek\Dataset\Training"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("--- Memuat Data Validasi ---")
# WAJIB: shuffle=False agar urutan gambar tidak diacak saat kita mencocokkan hasil tebakan vs kunci jawaban
val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    shuffle=False 
)

class_names = val_dataset.class_names

# Normalisasi data
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# 2. Load Model yang sudah dilatih
print("\n[INFO] Membaca model model_emotion_mobilenet.h5 ...")
model = tf.keras.models.load_model("model_emotion_mobilenet.h5")

# 3. Proses Prediksi
print("[INFO] Mengevaluasi model. Tunggu sebentar ya...")
# Mengambil kunci jawaban asli (y_true)
y_true = np.concatenate([y for x, y in val_dataset], axis=0)

# Meminta model menebak gambar (y_pred)
predictions = model.predict(normalized_val_dataset)
y_pred = np.argmax(predictions, axis=1)

# 4. Menampilkan Metrik Evaluasi (Accuracy, Precision, Recall, F1-Score)
print("\n" + "="*60)
print("             LAPORAN EVALUASI (CLASSIFICATION REPORT)")
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names))

# 5. Menampilkan Confusion Matrix
print("\n[INFO] Menampilkan grafik Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - MobileNetV2")
plt.ylabel("Label Asli (Kunci Jawaban)")
plt.xlabel("Label Prediksi (Tebakan Model)")
plt.show()