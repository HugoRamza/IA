#Importa las librerias CV
import cv2
import numpy as np
 
# Cargar la imagen
imagen = cv2.imread(r'C:\Users\Hugo\Downloads\lead17-2013-ford-mustang-v6-review.jpg')
# Obtener las dimensiones de la imagen
alto, ancho = imagen.shape[:2]
 
# Definir la matriz de transformación lineal (por ejemplo, escalar la imagen)
# Define la matriz de transformación lineal. En este caso, se utiliza una matriz de escala para reducir la imagen a la mitad (0.5x en ambas direcciones x e y):
escala_factor = 0.5
matriz_transformacion = np.float32([[escala_factor, 0, 0], [0, escala_factor, 0]])
 
# Aplicar la transformación lineal
imagen_transformada = cv2.warpAffine(imagen, matriz_transformacion, (ancho, alto))
 
# Mostrar la imagen original y la imagen transformada
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Imagen Transformada', imagen_transformada)
#Espera hasta que se presione una tecla y cierra las ventanas:
cv2.waitKey(0)
cv2.destroyAllWindows()