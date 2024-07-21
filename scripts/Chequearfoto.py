import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import random

class PhotoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Viewer")
        
        self.label = tk.Label(root, text="Seleccione una foto")
        self.label.pack(pady=10)
        
        self.button = tk.Button(root, text="Abrir Foto", command=self.open_photo)
        self.button.pack(pady=10)
        
        self.photo_label = tk.Label(root)
        self.photo_label.pack(pady=10)
        
        self.name_label = tk.Label(root, text="")
        self.name_label.pack(pady=10)
    
    def open_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            image = Image.open(file_path)
           # image = image.resize((400, 400), Image.ANTIALIAS)  # Redimensionar la imagen si es necesario
            photo = ImageTk.PhotoImage(image)
            
            self.photo_label.config(image=photo)
            self.photo_label.image = photo
            arreglo = ["Abel", "Carlos", "Federico G", "Federico R", "Florencia", "Franco A", "Franco S", "Gerard", "Gustavo", "Joaquin", "Juan", "Lautaro", "Lisandro", "Marco", "Matias", "Natalia", "Noelia", "Paola", "Victorio"]
            
            #cargar funciones para validar imagen
            
            
            # Supongamos que el nombre de la persona está en el nombre del archivo (sin la extensión)
            name = random.choice(arreglo)
            self.name_label.config(text=f"Nombre: {name}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoApp(root)
    root.mainloop()
