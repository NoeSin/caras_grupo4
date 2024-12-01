#importamos 
import imageio
import numpy as np


from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt #Para graficar
from skimage.io import imshow #Para graficar las imagenes
import tensorflow as tf


import os
import glob
from PIL import Image
import cv2

#import pyheif

import matplotlib.pyplot as plt

from PIL import Image



from Importar_Instalar import check_and_install

def nombres(directorio):
    """
    Obtenemos los nombres de las personas a partir de la estructura dle directorio, cada carpeta una persona
    """
    names = []
    for dir in os.listdir(directorio):
        names.append(dir)
    names.count
    return names


def convertir_heic_a_jpg(ruta_entrada, ruta_salida):
    """
    intenta convertir una imagen en Formato heic a jpg y 
    la almacena en la ruta de salida, con el mismo nombre de archivo.
    """
    
    check_and_install ("imageio") 
    try:
        # Lee la imagen HEIC
        imagen_heic = imageio.imread(ruta_entrada)  
        # Guarda la imagen como JPEG
        imageio.imwrite(ruta_salida, imagen_heic)
        print("La conversión se ha realizado con éxito.")
    except Exception as e:
        print(f"Error al convertir la imagen: {e}")

def convertir_persona_HIEC_a_jpg(ruta_persona):
    """
    para cada archivo, en la ruta recibida, con formato hiec crea uno con formato jpg en la misma ruta.
    
    """

    for archivo_hiec in glob.glob(os.path.join(ruta_persona, "*.HEIC")):
        
        HIEC_archivo_sin_ext, _ = os.path.splitext(archivo_hiec)
        jpg_path = HIEC_archivo_sin_ext + ".jpg"
        convertir_heic_a_jpg(archivo_hiec, jpg_path)     
        
               

import os
import glob
import cv2
import numpy as np
import pickle

def guardar_lote(nombre_persona, batch, indice_lote, directorio_destino):
    """
    Guarda un lote de imágenes procesadas en un archivo.

    Args:
        nombre_persona (str): Nombre de la persona asociada a las imágenes.
        batch (list): Lista de imágenes procesadas.
        indice_lote (int): Índice del lote actual.
        directorio_destino (str): Directorio donde se guardarán los lotes.
    """
    os.makedirs(directorio_destino, exist_ok=True)
    archivo_salida = os.path.join(directorio_destino, f"{nombre_persona}_lote_{indice_lote}.pkl")

    with open(archivo_salida, "wb") as f:
        pickle.dump(batch, f)

def leer_fotos_y_guardar_por_lotes(directorio_origen_f, directorio_destino, batch_size=100):
    """
    Lee fotos de cada persona por lotes y las guarda en archivos para optimizar memoria.

    Args:
        directorio_origen_f (str): Ruta del directorio que contiene subcarpetas con fotos organizadas por persona.
        directorio_destino (str): Ruta del directorio donde se guardarán los lotes procesados.
        batch_size (int): Número máximo de imágenes a procesar por lote.
    """
    for nombre_persona in os.listdir(directorio_origen_f):
        ruta_persona = os.path.join(directorio_origen_f, nombre_persona)

        if not os.path.isdir(ruta_persona):
            continue

        # Convertir HEIC a JPG si es necesario
        convertir_persona_HIEC_a_jpg(ruta_persona)

        # Recopilar las rutas de las imágenes
        rutas_imagenes = glob.glob(os.path.join(ruta_persona, "*.jp*g"))

        for i in range(0, len(rutas_imagenes), batch_size):
            batch_rutas = rutas_imagenes[i:i + batch_size]
            
            # Procesar solo el batch actual
            batch_imagenes = [cv2.imencode('.jpg', cv2.imread(ruta))[1] for ruta in batch_rutas]

            # Guardar el lote
            guardar_lote(nombre_persona, batch_imagenes, i // batch_size, directorio_destino)





def show_people(images):
    """
        Muestra una foto por persona.
        Recibe un diccionario con la persona como indice y una lista de sus fotos.
        
    """
    
    names_target = list(images.keys())

    rows = 5
    cols = 4
    fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))

    axarr=axarr.flatten()
    
    for image_index in range(len(names_target)):
        axarr[image_index].imshow(images[names_target[image_index]][0])
        axarr[image_index].set_xticks([])
        axarr[image_index].set_yticks([])
        axarr[image_index].set_title("Nombre:{}".format(names_target[image_index]))
    
    #Replico formato a los que no tienen caras
    for image_index in range(len(names_target), (rows * cols)):     
        axarr[image_index].set_xticks([])
        axarr[image_index].set_yticks([])
        axarr[image_index].set_title("Nombre:N/A")

    plt.suptitle("Las caras")
      



def show_people_agg_cant(dfotos, cant):
    
    """
        Muestra cantidad variable (recibida como parametro) de fotos por persona.
        Recibe un diccionario dfotos con la persona como indice y una lista de sus fotos.
        
    """
    #Visualizacion de "cant"  imagenes por cada alumno
    n_pics_per_person = cant
    n_cols = n_pics_per_person
    
    names_= list(dfotos.keys())
    n_rows = len(names_)  #18
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(1.6*n_cols, 2*n_rows))
    
    index=0
    for nombre in names_:
        
        # Select images using boolean indexing
        img = dfotos[nombre]
        
        for image_index in range(min(len(img),cant)):
            imag_i = img[image_index]
            axs[index, image_index].imshow(imag_i)
            axs[index, image_index].set_xticks([])
            axs[index, image_index].set_yticks([])
        
        axs[index, 0].set_title(nombre)
        index += 1




def matriz_fotos_desde_carpeta(dir_name_recorte ):
    import os
    # Tamaño fijo al que redimensionar todas las imágenes
    desired_size = (30, 30)
    # Listas para almacenar las imágenes y sus nombres
    images = [] #lista de fotos
    image_names = []
    image_person = [] #lista con los nombres de las personas de cada foto

    # Leer las imágenes del directorio y almacenarlas en las listas
    images = []
    for root, dirs, files in os.walk(dir_name_recorte):
        for dir_name in dirs:
            print("Carpeta:", dir_name)
            dir_path = os.path.join(root, dir_name) #directorio  de la persona

            for file_name in os.listdir(dir_path):

                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(dir_path, file_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
                if image is not None:
                    # Redimensionar la imagen al tamaño deseado
                    resized_image = cv2.resize(image, desired_size)

                    images.append(resized_image.flatten())  # Aplanar la imagen y agregarla a la lista
                    image_names.append(file_name)
                    image_person.append(dir_name)

    # Convertir la lista de imágenes a una matriz NumPy
    image_matrix = np.array(images) #matriz de fotos    
    return image_matrix




def recortar_imagen(image):
    import os
    import cv2
    import matplotlib.pyplot as plt
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Utilizar un clasificador específico para caras
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))

    # Recorrer las caras detectadas
    face_images = []
    for (x, y, w, h) in faces:
        # Recortar la cara de la imagen
        face_images.append(image[y:y+h, x:x+w])

    return face_images



def matriz_fotos_desde_carpeta(dir_name_recorte ):
    """
        Funcion, a partir de un directorio, busca cada carpeta en el directorio la interpreta como 
        el nombre de la persona,
        cada imagen dentro de esa carpeta la aplana, la reduce a 30*30 y la agrea a una matriz.
        La funcion devuelve image_matrix, la matriz  de todas las fotos vectorizadas 
        (al ser fotos de 30*30 se obtienen en un vector de 900 pixeles)
         tambien de vuelve un vector: image_person con el nombre de la persona de cada foto 
        (un item por cada fila de la matriz de fotos)
    """


    # Tamaño fijo al que redimensionar todas las imágenes
    desired_size = (30, 30)
    # Listas para almacenar las imágenes y sus nombres
    images = [] #lista de fotos
    image_names = []
    image_person = [] #lista con los nombres de las personas de cada foto

    # Leer las imágenes del directorio y almacenarlas en las listas
    images = []
    for root, dirs, files in os.walk(dir_name_recorte):
        for dir_name in dirs:
            print("Carpeta:", dir_name)
            dir_path = os.path.join(root, dir_name) #directorio  de la persona

            for file_name in os.listdir(dir_path):
            
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(dir_path, file_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
                if image is not None:
                    # Redimensionar la imagen al tamaño deseado
                    resized_image = cv2.resize(image, desired_size)
        
                    images.append(resized_image.flatten())  # Aplanar la imagen y agregarla a la lista
                    image_names.append(file_name)
                    image_person.append(dir_name)
                                   
    # Convertir la lista de imágenes a una matriz NumPy
    image_matrix = np.array(images) #matriz de fotos    
    return image_matrix, image_person



def Aplicar_PCA_Matriz_fotos(image_matrix, num_components, corrimiento =0):
    """
        Debe ingresar una matriz de con una Fila por Foto, en las columans los pixel de 900x900
        aplica PCA con la cantidad de componentes que recibe, y a partir de la componente indicada por el corrimiento
    """
    import os
    import cv2
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    imagenes_estandarizadas = scaler.fit_transform(image_matrix)
    pca             = PCA(n_components=num_components)
    pca.fit(imagenes_estandarizadas)
    # Proyectar las imágenes al espacio de los componentes principales
    projected_images = pca.transform(imagenes_estandarizadas)

    reduced_component = pca.components_.copy() # creamos un nuevo objeto con los componentes principales matrix (componenteppal*feature) (60*900)
    #reduced_component = reduced_component[corrimiento: +corrimiento]
    projected_images_reduced = imagenes_estandarizadas @ np.transpose(reduced_component) # imagenes_estandarizadas(cant_fotos,900)
                                                                                         # reduced_component(num_component,900) por eso hacemos la transpuesta
                                                                                         # projected image (cant_fotos,num_components)
   
   
   
    return projected_images_reduced




def transformar_imagen_pca(imagen, scaler, pca_components, num_componentes=50, inicio_componente=0):
    """
    # Función para transformar una imagen: 
    #   1: La escala segun el scaler recibido como parametro
    #   2: Aplica componentes recibidos como parametro, con un rango específico de componentes PCA recibido tambien en num_components
    """

    
    
    # Estandarizar la imagen
    imagen_estandarizada = scaler.transform(imagen)
    
    # Seleccionar el rango de componentes
    componentes_seleccionados = pca_components[inicio_componente:inicio_componente + num_componentes]
    
    # Realizar la proyección manualmente
    imagen_pca = np.dot(imagen_estandarizada, componentes_seleccionados.T)
    return imagen_pca




def transformar_imagen_pca(imagen, scaler, pca_components, num_componentes=50, inicio_componente=0):
    # Estandarizar la imagen
    imagen_estandarizada = scaler.transform(imagen)
    
    # Seleccionar el rango de componentes
    componentes_seleccionados = pca_components[inicio_componente:inicio_componente + num_componentes]
    
    # Realizar la proyección manualmente
    imagen_pca = np.dot(imagen_estandarizada, componentes_seleccionados.T)
    return imagen_pca


def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def relu_derivative(Z):
    return Z > 0

# Función de pérdida: entropía cruzada
def categorical_crossentropy(y_true, y_pred):
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss



# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2



# Backpropagation
def backpropagation(X, Y, Z1, A1, Z2, A2, W1, W2, b1, b2, learning_rate):
    m = X.shape[0]

    # Derivadas de la capa de salida
    dZ2 = A2 - Y.T
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # Derivadas de la capa oculta
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    # Actualización de los parámetros
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2



