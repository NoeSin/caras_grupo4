# Data Mining Avanzado. 
# Trabajo Práctico Final. Nuestras Caras

## Integrantes:
* Fierro, Abel
* Santesteban, Natalia
* Sincosky, Noelia

## Workflow.

### Opción1: Realizar todo el procesamiento completo.

__ 1. __ Ejecutar el script scripts\WF01_Aumentacion_imagenes.ipynb: Esta notebook genera mas imágenes a partir de las fotos de entrada almacenadas en "../Nuestras_caras" aplica cambios en el brillo,contraste,saturación y nitidez, filtros de enfoque y suavizado de bordes y además realiza una transformación de colores genera las imagenes y las almacena en "../Mas_fotos"

__ 2. __ Ejecutar el script scripts\WF02_Recortar_caras.ipynb: Esta recorre las nuevas imagenes generadas y aplica tranformaciones de formato si en necesario, además detecta caras y realiza recorte alrededor de ellas, este procedimiento lo realizamos a través de lotes por la cantidad de imagenes, las nuevas imagenes recortadas se almacenan en "../Caras_cortadas", cabe destacar que se detectan como si fuesen caras objetos que no lo son, en este caso nos tomamos el trabajo de revisar carpeta por carpeta borrando las que no son correctas.

__ 3. __ Ejecutar el script scripts\WF03_Pca_bp.ipynb: Esta recupera las imagenes en "../Caras_cortadas" aplica un pca con las 60 componentes a partir del elemento dos y guarda los elementos generados. Luego se aplica un modelo de red neuronal utilizando back propagation para realizar la clasificación y entrenar el modelo, recibe como entrada los componentes principales resultantes de PCA. Por último se almacenan los pesos y biases del modelo obtenido.

__ 4. __ Ejecutar por terminal scripts\WF04_Identificar_alumno.py: Este realiza la detección de cara, realiza transformaciones correspondientes y aplica pca de acuerdo a lo almacenado en los scripts anteriores, por último realiza la clasificación a través del modelo almacenado.

   __ --> 4.1 __ Hacer click en "Cargar foto" y subir la foto de la persona que quieres identificar.
   __ --> 4.2  __ Hacer click en "Quien es?" para saber el resultado.

   
### Opción2: Identificar el alumno correspondiente a una foto
  
   __ Ejecutar por terminal scripts\WF04_Identificar_alumno.py (Punto 4 de la opción1)

    __ --> 4.1 __ Hacer click en "Cargar foto" y subir la foto de la persona que quieres identificar.
    __--> 4.2 __ Hacer click en "Quien es?" para saber el resultado.
