import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuración
MODEL_PATH = "modelo_final_flora.keras"
CLASSES_PATH = "clases_flora.txt"
IMG_SIZE = (224, 224)

def load_model_and_classes():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'r') as f:
            class_names = []
            for line in f.readlines():
                name = line.strip()
                if '. ' in name:
                    name = name.split('. ', 1)[1]  # Quita número y punto inicial
                class_names.append(name)
        return model, class_names
    except Exception as e:
        print(f"Error al cargar modelo: {str(e)}")
        raise

def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error al procesar imagen: {str(e)}")
        return None

def predict_image(model, class_names, image_path):
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return None
    
    predictions = model.predict(processed_img)
    predicted_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_class = class_names[predicted_idx]
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'all_classes': class_names,
        'all_predictions': predictions[0]
    }

def main():
    model, class_names = load_model_and_classes()
    
    print(f"\n🔍 Modelo de clasificación con {len(class_names)} categorías:")
    print("Categorías disponibles:")
    for i, clase in enumerate(class_names, 1):
        print(f"{i}. {clase}")
    
    image_path = ("Cercos siliquastrum-6.png")
    
    if not os.path.exists(image_path):
        print("❌ Error: La imagen no existe")
        return
    
    result = predict_image(model, class_names, image_path)
    
    if result:
        print(f"\n📌 Resultado para la imagen '{os.path.basename(image_path)}':")
        print(f"La imagen es: {result['class']} (Confianza: {result['confidence']:.1%})")
        
        sorted_indices = np.argsort(result['all_predictions'])[::-1]
        print("\nTop 3 predicciones:")
        for idx in sorted_indices[:3]:
            print(f"- {class_names[idx]}: {result['all_predictions'][idx]:.1%}")

if __name__ == "__main__":
    main()
