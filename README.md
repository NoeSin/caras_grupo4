# caras_grupo4
## el archivo Funciones.py
Contiene Funciones utiles para el procesamiento de las imagenes.

## El archivo Recortar_Caras.ipynb
Recorre una carpeta dentro del proyecto la carpeta Nuestras Caras, que contiene una carpeta por alumno, y dentro de ella las imagenes del alumno, recorre esta carpeta y recorta armando imagenes en escala de grises conteniendo solo la cara del alumno, estas imagenes recortadas se guardan en la carpeta Caras_cortadas con la misma estructura de una carpeta por persona.

Carpeta de origen, dentro de proyecto las fotos iniciales se encuentran en la Carpeta Nuestras caras
    Variables importantes:
        Nombres de las personas: nombres_personas
        Diccionario con fotos originales : fotos (indice persona, lista de fotos)
            Cantidad de personas: 19
            Cantidad de fotos por persona:
            - Abel: 15
            - Carlos: 16
            - Federico G: 17
            - Federico R: 19
            - Florencia: 12
            - Franco A: 19
            - Franco S: 10
            - Gerard: 13
            - Gustavo: 12
            - Joaquin: 13
            - Juan: 14
            - Lautaro: 20
            - Lisandro: 18
            - Marco: 11
            - Matias: 17
            - Natalia: 13
            - Noelia: 13
            - Paola: 18
            - Victorio: 11


    Fotos recortadas en la carpeta Caras_cortadas con la misma estructura interna de una carpeta por persona:
    Cantidad de personas: 19
        Cantidad de fotos por persona:
            - Abel: 13
            - Carlos: 15
            - Federico G: 17
            - Federico R: 19
            - Florencia: 12
            - Franco A: 16
            - Franco S: 10
            - Gerard: 13
            - Gustavo: 11
            - Joaquin: 13
            - Juan: 15
            - Lautaro: 20
            - Lisandro: 17
            - Marco: 11
            - Matias: 17
            - Natalia: 13
            - Noelia: 11
            - Paola: 18
            - Victorio: 11

## El notebook PCA-Caras
Obtinene las imagenes de caras ya recortadas y en  escala de grises de la carpeta ../Caras_cortadas

Convertimos las imágenes a un Numpy Array. Conversión de los datos a una matriz que contiene datos sin procesar. La función array by numpy toma una lista como entrada.
Redimensionamos las imagenes a 30 X 30
La imagen se agrega a un array con los 900 pixel
Cada imagen vectorizada se agraga a una matriz formando una matriz de dimensiones cantidad de imagen X 900 pixel

__variables:__
* __names__ : array con el nombre de las personas
* __image_matrix__: matriz de (cantFotos X 900) contiene un renglon por cada foto aplanada en un vector.
*__projected_images__: las imagenes proyectadas por las 900 componentes pca.
*__

Ademas en este notebook se realiza PCA para obtener las componentes principals de las fotos.
Esto se obtiene de dos maneras,  paso a paso con multiplicacion de matrices y otra de las formas fue utilizando la libreria sklearn.

Las imagenes se escalan para introducirlas a PCA, almacenamos tanto el esccalador como las componentes obtenidas estos no servira para tratar cualquier imagen a entrenar en la red .

__Guardamos:__
    escaler:  ../scaler.pkl
    componentes: ../Componentes_pca.npy
    
Pca libreria 

Pca manual
 
