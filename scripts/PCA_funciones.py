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