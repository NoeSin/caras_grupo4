# Data Mining Avanzado. 
# Trabajo Práctico Final. Nuestras Caras

## Integrantes:
* Fierro, Abel
* Santesteban, Natalia
* Sincosky, Noelia

## Workflow.

### Opción1: Realizar todo el procesamiento completo.

__1.__ Ejecutar el script scripts\WF01_Aumentacion_imagenes.ipynb: Esta notebook genera más imágenes a partir de las fotos de entrada almacenadas en "../Nuestras_caras" y aplica cambios en el brillo, contraste, saturación y nitidez, además incorpora filtros de enfoque y suavizado de bordes y realiza una transformación de colores, genera las imágenes y las almacena en "../Mas_fotos"

__2.__ Ejecutar el script scripts\WF02_Recortar_caras.ipynb: Esta recorre las nuevas imágenes generadas y aplica transformaciones de formato si en necesario, además detecta caras y realiza un recorte alrededor de ellas. Este procedimiento lo realizamos a través de lotes por la cantidad de imagenes. Las nuevas imagenes recortadas se almacenan en "../Caras_cortadas". Es importante destacar que se detectan como si fuesen caras objetos que no lo son, en este caso nos tomamos el trabajo de revisar carpeta por carpeta borrando las que no son correctas.

__3.__ Ejecutar el script scripts\WF03_Pca_bp.ipynb: Esta recupera las imagenes en "../Caras_cortadas", aplica un pca con las 60 componentes a partir del elemento dos y guarda los elementos generados. Luego se aplica un modelo de red neuronal utilizando back propagation para realizar la clasificación y entrenar el modelo. Este recibe como entrada los componentes principales resultantes de PCA y por último se almacenan los pesos y bases del modelo obtenido.

__4.__ Ejecutar por terminal scripts\WF04_Identificar_alumno.py: Este script realiza la detección de cara. Ejecuta las transformaciones correspondientes y aplica pca de acuerdo a lo almacenado en los scripts anteriores. Por último realiza la clasificación a través del modelo almacenado.


   __4.1__ Hacer click en "Cargar foto" y subir la foto de la persona que quieres identificar.

   
   __4.2__ Hacer click en "Quien es?" para saber el resultado.

   
### Opción2: Identificar el alumno correspondiente a una foto
   Ejecutar por terminal scripts\WF04_Identificar_alumno.py (Punto 4 de la opción1).

   __1.__ Hacer click en "Cargar foto" y subir la foto de la persona que quieres identificar.
   
   __2.__ Hacer click en "Quien es?" para saber el resultado.
 
