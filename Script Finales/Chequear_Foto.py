import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import pickle
from Funciones import recortar_imagen
from Funciones import relu, softmax, relu_derivative, categorical_crossentropy, forward_propagation, backpropagation
from PCA_funciones import transformar_imagen_pca
import numpy as np
import cv2 as cv
import os


# Función para cargar una foto
def cargar_foto():
    global foto, img_label, img, img_procesada_label
    archivo = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg")])
    if archivo:
        img = Image.open(archivo)
        imgr = img.resize((300, 300))  # Redimensionar la imagen para mostrarla
        foto = ImageTk.PhotoImage(imgr)
        img_label.config(image=foto)
        img_label.image = foto

        img = np.array(img)

        # Limpiar cualquier imagen procesada previa
        img_procesada_label.config(image="")
        img_procesada_label.image = None

        # Activar el botón de procesar y desactivar el de cargar
        boton_procesar.config(state="normal")
        boton_cargar.config(state="disabled")
        resultado_label.config(text="")


# Función de procesamiento de la foto
def quien_es():
    try:
        # Cargar los componentes PCA guardados
        ruta_pca = "../PCA/Componentes_pca.npz"
        data = np.load(ruta_pca, allow_pickle=True)
        components = data['components']
        mean = data['mean']
        labels = data['labels']

        # Cargar el escalador
        ruta_scaler = '../PCA/scaler.pkl'
        with open(ruta_scaler, 'rb') as f:
            scaler = pickle.load(f)

        # Recortar caras de la imagen cargada
        face_images = recortar_imagen(img)

        if not face_images:  # Si no se detectan caras
            resultado_label.config(text="No se detectaron caras en la imagen.")
            boton_procesar.config(state="disabled")
            boton_cargar.config(state="normal")
            return

        for face_image in face_images:
            nueva_imagen = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
            nueva_imagen = cv.resize(nueva_imagen, (30, 30))
            nueva_imagen = nueva_imagen.flatten()

            # Mostrar la imagen procesada en la interfaz
            imagen_procesada = Image.fromarray(nueva_imagen.reshape(30, 30))  # Reconstruir a 30x30
            imagen_procesada = imagen_procesada.resize((150, 150))  # Redimensionar para visualización
            foto_procesada = ImageTk.PhotoImage(imagen_procesada)
            img_procesada_label.config(image=foto_procesada)
            img_procesada_label.image = foto_procesada

            lista = []
            lista.append(nueva_imagen)
            image_matrix = np.array(lista)

        # Cargar los pesos de la red neuronal
        ruta_modelo = "../Back_propagation/Modelo"
        W1 = np.load(os.path.join(ruta_modelo, "W1.npy"))
        b1 = np.load(os.path.join(ruta_modelo, "b1.npy"))
        W2 = np.load(os.path.join(ruta_modelo, "W2.npy"))
        b2 = np.load(os.path.join(ruta_modelo, "b2.npy"))

        # Estandarizar las imágenes usando el escalador cargado
        nuevas_imagenes_estandarizadas = scaler.transform(image_matrix)

        # Proyectar al espacio PCA
        imagen_pca_reducida = np.dot(nuevas_imagenes_estandarizadas, components.T)

        # Forward propagation
        Z1_test1, A1_test1, Z2_test1, A2_test1 = forward_propagation(imagen_pca_reducida, W1, b1, W2, b2)
        prediction = A2_test1.T[0]

        predicted_label = int(np.argmax(prediction))
        predicted_probability = prediction[predicted_label]

        # Mostrar el resultado
        resultado_label.config(text=f"{labels[predicted_label]} (Probabilidad: {predicted_probability:.2f})")

    except FileNotFoundError as e:
        resultado_label.config(text="Error: No se encontró el archivo necesario.")
        print(f"Error: {e}")
    except UnboundLocalError as e:
        resultado_label.config(text="Error: Imagen no encontrada o no cargada.")
        print(f"Error: {e}")
    except Exception as e:
        resultado_label.config(text="Ha ocurrido un error inesperado.")
        print(f"Error inesperado: {e}")

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
img_label.pack(side=tk.LEFT, padx=10)

# Etiqueta para mostrar la imagen procesada
img_procesada_label = Label(ventana)
img_procesada_label.pack(side=tk.RIGHT, padx=10)

# Etiqueta para mostrar el resultado
resultado_label = Label(ventana, text="", font=("Helvetica", 14))
resultado_label.pack(pady=10)

# Ejecutar la aplicación
ventana.mainloop()
