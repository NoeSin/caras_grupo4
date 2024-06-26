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
        
               

def Leer_fotos(directorio_origen_f):
    """
        Lee la ruta lee interpretando que cada carpeta es una persona, y dentro de esa careta se encuentran las fotos de esa persona.
        Devuelve un diccionario con el nombre de la pesrona como índice y una lista de sus fotos.
    """

    # Diccionario para almacenar las fotos
    fotos_int= {} # diccionario interno q sea persona y el vector de fotos, 

    # Recorrer las subcarpetas

    for nombre_persona in os.listdir(directorio_origen_f):
        # Ruta de la subcarpeta
        ruta_persona = os.path.join(directorio_origen_f, nombre_persona)

        # Lista para almacenar las fotos de la persona
        fotos_persona = []    
        #convertir hiec a png
        convertir_persona_HIEC_a_jpg(ruta_persona)
        # Recorrer las fotos de la persona
        for archivo in glob.glob(os.path.join(ruta_persona, "*.jpg")) :
            # Cargar la imagen
            imagen = cv2.imread(archivo)              
            fotos_persona.append(imagen)             
        fotos_int[nombre_persona] = fotos_persona
        
    return fotos_int



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
