# Manual de Usuario

## 1. Acceso a la aplicación

Una vez que la aplicación ha sido desplegado, puedes acceder al dashboard desde cualquier navegador.

* **Cuando estás en la misma red/computador:** Abre tu navegador y visita `http://localhost:8001`
* **Cuando el sistema estás en la nube (AWS):** Visita la dirección IP pública proporcionada por AWS (ejemplo: `http://IP:8001`).

Al ingresar, deberías ver de inmediato el dashboard principal de la aplicación.

---

## 2. Guía de Uso Paso a Paso

### Paso 1: Carga del Audio

El sistema está diseñado para procesar archivos de formato **WAV** (`.wav`).

1. Ubica la sección **"Subir Archivo de Audio"** en el panel principal.
2. Haz clic en el botón gris **"Seleccionar Archivo"**.
3. Navega por tus carpetas y selecciona la grabación de voz.
4. Una vez seleccionado, el nombre del archivo aparecerá en la barra blanca de la derecha, confirmando que está listo.
![alt text](image-1.png)

### Paso 2: Procesamiento

1. Haz clic en el botón naranja **"INICIAR ANÁLISIS"**.
2. El botón cambia a estado de espera ("Procesando..."). En este momento, la aplicación está enviando el audio al servidor para extraer sus características acústicas y pasarlo por la red neuronal.
3. Espera unos segundos hasta que el dashboard muestre los resultados.

### Paso 3: Visualización

El dashboard se desplazará hacia abajo para mostrar el **Panel de Resultados**, el cual está dividido en tres áreas.

---![alt text](image.png)

## 3. Interpretación de los Resultados

### A. Emoción Predominante

Este panel muestra el resultado principal del modelo.

* **Emoción:** Indica el estado de ánimo principal detectado (Alegría, Tristeza, Enojo, Miedo o Neutral) acompañado de un emoji.
* **Precisión (%):** Indica el nivel de precisión del modelo sobre su predicción.

### B. Espectrograma de Frecuencias

Muestra un espectrograma del audio. Esta imagen es la representación visual del sonido a lo largo del tiempo, mostrando la intensidad de las frecuencias.

### C. Distribución de Probabilidades

Dado que la voz humana puede mezclar emociones, este gráfico de barras muestra los porcentajes distribuidos entre las 5 categorías evaluadas, permitiéndole entender si, por ejemplo, además de la emoción detectada, hubo rastros de otra emoción en la voz.

---

## 4. Preguntas Frecuentes y Solución de Errores

* **No puedo seleccionar mi archivo de audio:**
  La aplicación solo permite seleccionar archivos con extensión `.wav`. Si tu audio está en otro formato, debes convertirlo usando una herramienta gratuita en línea antes de subirlo.

* **El botón se queda cargando y nunca da resultado:**
  Esto indica una pérdida de comunicación con el servidor. Refresca la página web (F5) y vuelve a intentarlo. Si el error persiste, verifica que el servidor de AWS esté encendido.

* **El modelo muestra baja confianza (ej. 30%):**
  Esto sucede cuando el audio tiene mucho ruido de fondo, silencios prolongados o la emoción expresada es muy ambigua. Intenta subir un audio grabado en un ambiente más silencioso.
