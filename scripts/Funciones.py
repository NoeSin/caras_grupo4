
import numpy as np


from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt #Para graficar
from skimage.io import imshow #Para graficar las imagenes
import tensorflow as tf


import os
import glob
#from PIL import Image
import cv2

#import pyheif

import matplotlib.pyplot as plt

from PIL import Image
import imageio


def nombres(directorio):
    names = []
    for dir in os.listdir(directorio):
        names.append(dir)
    names.count
    return names



def convertir_heic_a_jpg(ruta_entrada, ruta_salida):
    try:
        # Lee la imagen HEIC
        imagen_heic = imageio.imread(ruta_entrada)
        # Guarda la imagen como JPEG
        imageio.imwrite(ruta_salida, imagen_heic)
        print("La conversión se ha realizado con éxito.")
    except Exception as e:
        print(f"Error al convertir la imagen: {e}")

def convertir_persona_HIEC_a_jpg(ruta_persona):

    for archivo_hiec in glob.glob(os.path.join(ruta_persona, "*.HEIC")):
        
        HIEC_archivo_sin_ext, _ = os.path.splitext(archivo_hiec)
        jpg_path = HIEC_archivo_sin_ext + ".jpg"
        convertir_heic_a_jpg(archivo_hiec, jpg_path)       
        
               

def Leer_fotos(directorio_origen_f):
    # Diccionario para almacenar las fotos
    fotos_int= {} # diccionario interno q sea persona y el vector de fotos, 
    #despues elegimos una de las dos estructuras
    #v_fotos_personas =[] #este vector tendra todas las fotos
    #v_nombre_personas = [] #este vector tendra el nombre de la persona, forrespondiente a la foto del vetor fotos_persona en la misma posicion
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
            #v_fotos_personas.append(imagen)
            #_nombre_personas.append(nombre_persona)
            # Agregar las fotos de la persona al diccionario
        fotos_int[nombre_persona] = fotos_persona
    return fotos_int




def show_people(images):
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
