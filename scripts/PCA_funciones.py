import numpy as np
import os
import cv2
import numpy as np

def aplicar_dct(imagen):
    """
    Aplica la Transformada Discreta del Coseno (DCT) a la imagen.
    Devuelve la imagen transformada en el dominio de frecuencias.
    """
    return cv2.dct(np.float32(imagen))

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
                    #
                    #resized_image = aplicar_dct(resized_image)
                    #sobel_x = cv2.Sobel(resized_image, cv2.CV_64F, 1, 0, ksize=3)
                    #sobel_y = cv2.Sobel(resized_image, cv2.CV_64F, 0, 1, ksize=3)
                    #sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)  # Magnitud combinada de Sobel

                    # Laplaciano: Detecta bordes enfatizando cambios bruscos de intensidad
                    #laplacian = cv2.Laplacian(resized_image, cv2.CV_64F, ksize=3)

                    # Combinar Sobel y Laplaciano para crear una imagen mejorada
                    #resized_image = cv2.convertScaleAbs(sobel_magnitude + laplacian)
                    
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

def matriz_fotos_nuevas(dir_name):
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
