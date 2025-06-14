import tensorflow as tf
import os
import numpy as np

# Configuración
DATASET_DIR = "Flora_dataset_final"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
FINE_TUNE_EPOCHS = 10

# Aumentos de datos activados solo en entrenamiento
def load_and_preprocess_image(path, label, augment=False):
    try:
        image = tf.io.read_file(path)
        if tf.strings.regex_full_match(path, r".*\.jpg$") or tf.strings.regex_full_match(path, r".*\.jpeg$"):
            image = tf.image.decode_jpeg(image, channels=3)
        else:
            image = tf.image.decode_png(image, channels=3)

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, IMG_SIZE)

        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)

        return image, label
    except Exception as e:
        print(f"❌ Error procesando imagen {path.numpy().decode('utf-8')}: {str(e)}")
        return None, None

# Crear dataset completo
def create_dataset(directory):
    class_names = sorted(
        [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )

    file_paths, labels = [], []
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        count = 0
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(class_dir, file))
                labels.append(label)
                count += 1
        print(f"📂 {class_name}: {count} imágenes")

    file_paths = tf.convert_to_tensor(file_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    return ds, class_names

# Cargar dataset
print("⏳ Cargando dataset...")
try:
    full_ds, class_names = create_dataset(DATASET_DIR)
    num_classes = len(class_names)
    dataset_size = len(list(full_ds))
    if dataset_size == 0:
        raise ValueError("Dataset vacío")
    print(f"✅ {dataset_size} imágenes totales en {num_classes} clases")
except Exception as e:
    print(f"❌ Error al cargar dataset: {str(e)}")
    exit()

# Dividir en entrenamiento y validación
train_size = int(0.8 * dataset_size)
train_ds = full_ds.take(train_size).map(
    lambda x, y: load_and_preprocess_image(x, y, augment=True), num_parallel_calls=tf.data.AUTOTUNE
).shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = full_ds.skip(train_size).map(
    lambda x, y: load_and_preprocess_image(x, y, augment=False), num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# MODELO con TRANSFER LEARNING
print("🧠 Cargando MobileNetV2 preentrenada...")
base_model = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3),
                                                include_top=False,
                                                weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("modelo_entrenado.keras", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

# ENTRENAMIENTO INICIAL
print("🚀 Entrenando modelo base...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# FINE-TUNING
print("🔧 Activando fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("🔁 Entrenando fine-tuning...")
history_fine = model.fit(train_ds,
                         validation_data=val_ds,
                         epochs=EPOCHS + FINE_TUNE_EPOCHS,
                         initial_epoch=history.epoch[-1] + 1,
                         callbacks=callbacks)

# GUARDADO FINAL
print("💾 Guardando modelo y clases...")
model.save("modelo_entrenado.keras")

with open("clases_flora.txt", "w") as f:
    for i, name in enumerate(class_names):
        f.write(f"{i}. {name}\n")

print(f"\n🎉 Entrenamiento completo. Modelo: modelo_entrenado.keras | Clases: {num_classes}")
