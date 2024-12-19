import cv2
import os

# Ruta de la carpeta de entrada y salida
input_folder = r'input'
output_folder = r'output'

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Función para segmentar la imagen y mejorar los detalles internos
def segment_image(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar ecualización de histograma para mejorar el contraste
    equalized = cv2.equalizeHist(gray)
    
    # Aplicar un umbral para segmentar la imagen
    _, segmented = cv2.threshold(equalized, 5, 255, cv2.THRESH_BINARY)
    
    # Aplicar un filtro de detección de bordes para resaltar los detalles internos
    edges = cv2.Canny(equalized, 200, 255)
    
    # Combinar la imagen segmentada con los bordes detectados
    combined = cv2.bitwise_or(segmented, edges)
    
    return combined

# Procesar cada imagen en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Leer la imagen
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        # Segmentar la imagen y mejorar los detalles internos
        segmented_image = segment_image(image)
        
        # Guardar la imagen segmentada en la carpeta de salida
        output_path = os.path.join(output_folder, 'segmented_' + filename)
        cv2.imwrite(output_path, segmented_image)

print('Segmentación completada.')