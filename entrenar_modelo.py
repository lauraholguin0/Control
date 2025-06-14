import tensorflow as tf
import os
import math
import numpy as np
from PIL import Image

# Configuración
DATASET_DIR = "Flora_dataset_final"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Reducido para manejar datasets más pequeños
EPOCHS = 20

# 1. Función mejorada para cargar y preprocesar imágenes
def load_and_preprocess_image(path, label):
    try:
        # Leer el archivo de imagen
        image = tf.io.read_file(path)
        
        # Detectar y manejar diferentes formatos de imagen
        if tf.strings.regex_full_match(path, ".*\.jpg$") or tf.strings.regex_full_match(path, ".*\.jpeg$"):
            image = tf.image.decode_jpeg(image, channels=3)
        else:
            image = tf.image.decode_png(image, channels=3)
        
        # Convertir a float32 y normalizar
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, IMG_SIZE)
        
        return image, label
    except Exception as e:
        print(f"Error procesando imagen {path.numpy().decode('utf-8')}: {str(e)}")
        return None, None

# 2. Cargar dataset con manejo de orden no secuencial
def create_dataset(directory):
    # Obtener clases y ordenarlas naturalmente (1, 2, 3...)
    class_names = sorted(
        [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )
    
    file_paths = []
    labels = []
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        images_in_class = 0
        
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(class_dir, file))
                labels.append(label)
                images_in_class += 1
        
        print(f"📂 {class_name}: {images_in_class} imágenes")
    
    # Convertir a tensores
    file_paths = tf.convert_to_tensor(file_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    
    # Crear dataset
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # Mapear y filtrar
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x, y: x is not None)  # Filtrar imágenes con errores
    
    return ds, class_names

# 3. Cargar y preparar datos
print("⏳ Cargando y analizando dataset...")
try:
    full_ds, class_names = create_dataset(DATASET_DIR)
    num_classes = len(class_names)
    
    # Calcular tamaño del dataset
    dataset_size = len(list(full_ds))
    if dataset_size == 0:
        raise ValueError("No se encontraron imágenes válidas")
    
    print(f"✅ Dataset cargado: {dataset_size} imágenes en {num_classes} categorías")
    
    # Dividir en entrenamiento (80%) y validación (20%)
    train_size = int(0.8 * dataset_size)
    val_ds = full_ds.skip(train_size).batch(BATCH_SIZE)
    train_ds = full_ds.take(train_size).shuffle(buffer_size=1000).batch(BATCH_SIZE)
    
    # Optimización
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
except Exception as e:
    print(f"❌ Error fatal: {str(e)}")
    exit()

# 4. Construir modelo mejorado
print("\n🔨 Construyendo modelo CNN...")
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(*IMG_SIZE, 3)),
    
    # Bloque 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    
    # Bloque 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
    
    # Bloque 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.4),
    
    # Clasificador
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 5. Compilar modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Entrenamiento con callbacks
print(f"\n🎯 Comenzando entrenamiento con {dataset_size} imágenes...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'mejor_modelo_flora.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-6
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# 7. Guardar resultados
print("\n💾 Guardando resultados...")
model.save("modelo_final_flora.keras")

with open("clases_flora.txt", "w") as f:
    for i, clase in enumerate(class_names):
        f.write(f"{i}. {clase}\n")

# Generar resumen
print("\n📊 Resumen final:")
print(f"- Total imágenes: {dataset_size}")
print(f"- Clases: {', '.join(class_names)}")
print(f"- Mejor val_accuracy: {max(history.history['val_accuracy']):.2%}")
print(f"- Modelo guardado en: modelo_final_flora.keras")
print("🎉 ¡Entrenamiento completado con éxito!")
