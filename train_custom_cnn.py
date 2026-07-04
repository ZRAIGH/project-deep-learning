import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers, models, optimizers

DATASET_DIR = r"C:\Users\zora\OneDrive\Documents\project-deep-learning\Dataset\Training"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# 1. Load Data
train_ds = image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", seed=123,
    shuffle=True, image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb"
)
val_ds = image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
    shuffle=True, image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb"
)

num_classes = len(train_ds.class_names)

# Data Augmentasi & Normalisasi Khusus Custom CNN (0 sampai 1)
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
])

rescale_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (rescale_layer(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y)).prefetch(tf.data.AUTOTUNE)

# 2. Bangun Model Custom CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = tf.keras.callbacks.ModelCheckpoint("model_emotion_custom_cnn.h5", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1)

print("\n[INFO] Melatih Model Pembanding (Custom CNN)...")
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[checkpoint, reduce_lr])
print("\n[SUKSES] model_emotion_custom_cnn.h5 berhasil disimpan!")