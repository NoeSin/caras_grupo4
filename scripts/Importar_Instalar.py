import subprocess
import sys

def check_and_install(package):
    try:
        # Intentar importar el paquete
        __import__(package)
    except ImportError:
        # Si falla la importación, instalar el paquete usando pip
        print(f"{package} no está instalado. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


