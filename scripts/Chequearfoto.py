import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import pickle
from Funciones import recortar_imagen
import numpy as np
import cv2 as cv
import os
# Función para transformar una imagen con un rango específico de componentes PCA
def transformar_imagen_pca(imagen, scaler, pca_components, num_componentes=50, inicio_componente=0):
    # Estandarizar la imagen
    imagen_estandarizada = scaler.transform(imagen)
    
    # Seleccionar el rango de componentes
    componentes_seleccionados = pca_components[inicio_componente:inicio_componente + num_componentes]
    
    # Realizar la proyección manualmente
    imagen_pca = np.dot(imagen_estandarizada, componentes_seleccionados.T)
    return imagen_pca

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


# Funciones de activación
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



# Función para cargar una foto
def cargar_foto():
    global foto, img_label,img
    archivo = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg")])
    if archivo:
        img = Image.open(archivo)
        imgr = img.resize((300, 300))  # Redimensionar la imagen para mostrarla
        foto = ImageTk.PhotoImage(imgr)
        img_label.config(image=foto)
        img_label.image = foto
        
        img = np.array(img)
        
        # Activar el botón de procesar y desactivar el de cargar
        boton_procesar.config(state="normal")
        boton_cargar.config(state="disabled")
        resultado_label.config(text="")
        #return img
# Función de procesamiento de la foto
def quien_es():
    # pca cargar modelo y procesar
    

    # Cargar los componentes PCA guardados
    ruta_pca = "../PCA/Componentes_pca.npy" #lo tenemos en memoria pero lo leemos nuevamente por si separamos el codigo
    pca_components = np.load(ruta_pca)

    # Cargar el escalador
    ruta_scaler ='../PCA/scaler.pkl' #
    with open(ruta_scaler, 'rb') as f:
        scaler = pickle.load(f)
    num_componentes = 60
    inicio_componente = 2 #inicia a partir de la 3ta componente
    face_images = recortar_imagen(img)
    # images.extend(face_images)
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Utilizar un clasificador específico para caras
    #face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")

    # Detectar rostros en la imagen
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))

    # Recorrer las caras detectadas
    #face_images = []
    #for (x, y, w, h) in faces:
        # Recortar la cara de la imagen
     #   face_images.append(image[y:y+h, x:x+w])
        
        
    for face_image in face_images:
        nueva_imagen = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)

        # Redimensionar la imagen
        nueva_imagen = cv.resize(nueva_imagen, (30, 30))
        # Aplanar la imagen
        nueva_imagen = nueva_imagen.flatten()

        # Aplicar PCA
        lista=[]
        lista.append(nueva_imagen)
        image_matrix = np.array(lista) 
    #cargar pesos
    ruta_modelo = "../Back_propagation/Modelo"
    W1 = np.load(os.path.join(ruta_modelo, "W1.npy"))
    b1 = np.load(os.path.join(ruta_modelo, "b1.npy"))
    W2 = np.load(os.path.join(ruta_modelo, "W2.npy"))
    b2 = np.load(os.path.join(ruta_modelo, "b2.npy"))
    
    # Transformar la nueva imagen utilizando los componentes especificados
    imagen_pca_reducida = transformar_imagen_pca( image_matrix, scaler, pca_components, num_componentes, inicio_componente)
    Z1_test1, A1_test1, Z2_tes1t, A2_test1 = forward_propagation(imagen_pca_reducida, W1, b1, W2, b2)
    
    prediction =  A2_test1.T[0]
    
    predicted_label = int(np.argmax(prediction))
    
    
    #print(predicted_label)
    Alumnos=['ABEL','CARLOS','FEDE G','FEDE R','FLOR','FRANCO A','FRANCO S','GERARD','GUSTAVO','JOAQUIN','JUAN','LAUTI','LISO','MARC','MATI','NATI','NOE','PAO','VIC']
    
    resultado_label.config(text=Alumnos[predicted_label])
    
    # Desactivar el botón de procesar y activar el de cargar
    boton_procesar.config(state="disabled")
    boton_cargar.config(state="normal")

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("App de reconocimiento")

# Botón para cargar una foto
boton_cargar = tk.Button(ventana, text="Cargar Foto", command=cargar_foto)

boton_cargar.pack(pady=10)

# Botón para procesar la foto
boton_procesar = tk.Button(ventana, text="¿Quién es?", command=quien_es, state="disabled")
boton_procesar.pack(pady=10)

# Etiqueta para mostrar la imagen cargada
img_label = Label(ventana)
img_label.pack()

# Etiqueta para mostrar el resultado
resultado_label = Label(ventana, text="", font=("Helvetica", 14))
resultado_label.pack(pady=10)

# Ejecutar la aplicación
ventana.mainloop()
