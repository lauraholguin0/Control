import os
import shutil

# Ruta original donde están las carpetas 'arboles', 'flores', 'arbustos'
origen_base = "Flora_parque_ambrosio"
# Nueva carpeta organizada
destino_base = "Flora_dataset_final"

os.makedirs(destino_base, exist_ok=True)

# Recorremos los grupos (arboles, arbustos, flores)
for grupo in os.listdir(origen_base):
    ruta_grupo = os.path.join(origen_base, grupo)
    
    if os.path.isdir(ruta_grupo):
        for subclase in os.listdir(ruta_grupo):
            ruta_subclase = os.path.join(ruta_grupo, subclase)

            if os.path.isdir(ruta_subclase):
                ruta_destino_subclase = os.path.join(destino_base, subclase)
                os.makedirs(ruta_destino_subclase, exist_ok=True)

                for archivo in os.listdir(ruta_subclase):
                    if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                        origen = os.path.join(ruta_subclase, archivo)
                        destino = os.path.join(ruta_destino_subclase, archivo)
                        shutil.copy(origen, destino)

print("✅ Reorganización completa: subclases ahora son clases principales.")
