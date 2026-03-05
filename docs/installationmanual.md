# Manual de  Instalación

## 1. Requisitos Previos

* **Python 3.10+** y `pip`.

* **FFmpeg**: Vital para el procesamiento de señales digitales con Librosa.

* **Docker**: Para despliegues rápidos y consistentes.

* **Git / Wget**: Para el control de versiones y descarga del código.
* **AWS CLI:** Configurado con credenciales (aws configure).

* **Bucket S3:** Creado en AWS para almacenar los datos del proyecto.

* **Usuario IAM:** Con permisos de acceso configurados para leer/escribir en el bucket S3.

## 2. Instalación Local

Si deseas probar el código en tu computadora sin contenedores:

1. **Clonar el repositorio y entrar a la subcarpeta de la app:**

   ```bash
   git clone https://github.com/jessicagu039/clasificacion-emociones-audio.git

   cd clasificacion-emociones-audio/tablero
  

2. **Configurar credenciales y descargar datos del MESD**

    ```bash
    aws configure
    aws s3 sync s3://amzn-s3-maia-mesd-2026/raw/all-wavs/MexicanEmotionalSpeechDatabase/ ./datos_s3
    ```

3. **Entorno virtual y dependencias**

    ```bash
    python -m venv venv
    source venv/bin/activate  
    pip install -r requirements.txt
    ```

4. **Ejecución**

    ```bash
    python app.py
    ```

   Accede a: `http://localhost:8001`

## 3. Despliegue en AWS EC2

### A. Configuración Local (Sin Docker)

 1. **Conexión SSH a la instancia EC2**

```bash
ssh -i "llaves_aws.pem" ubuntu@[IP_ADDRESS]
```

 1. **Clonar el repositorio y entrar a la carpeta del tablero**

```bash
git clone https://github.com/jessicagu039/clasificacion-emociones-audio.git
cd clasificacion-emociones-audio/tablero
```

 1. **Configurar AWS CLI**

```bash
aws configure
# Ingresa tu Access Key ID, Secret Access Key, región y formato de salida
```

 1. **Crear carpeta para el modelo y descargar los pesos .h5**

```bash
mkdir -p model
wget -O model/best_model.h5 https://raw.githubusercontent.com/jessicagu039/clasificacion-emociones-audio/main/modelos/best_model.h5
```

 1. **Descargar datos del dataset MESD desde S3**

```bash
aws s3 sync s3://amzn-s3-maia-mesd-2026/raw/all-wavs/MexicanEmotionalSpeechDatabase/ ./datos_s3
```

 1. **Construir la imagen de Docker**

```bash
docker build -t emotion-app .
```

 1. **Ejecutar el contenedor**

```bash
docker run -d -p 8001:8001 --name emotion_container \
  -v /ruta/a/tu/datos_s3:/app/datos_s3 \
  -v /ruta/a/tu/model:/app/model \
  emotion-app
```

 1. **Verificar que el contenedor esté funcionando**

```bash
docker logs emotion_container
```

### B. Despliegue con Docker Compose (Recomendado)

Si ya tienes Docker y Docker Compose instalados en tu máquina local o en EC2, sigue este método:

1. **Clonar el repositorio:**

   ```bash
   git clone https://github.com/jessicagu039/clasificacion-emociones-audio.git
   cd clasificacion-emociones-audio/tablero
   ```

2. **Configurar AWS CLI y descargar datos (si no lo has hecho):**

   ```bash
   aws configure
   aws s3 sync s3://amzn-s3-maia-mesd-2026/raw/all-wavs/MexicanEmotionalSpeechDatabase/ ./datos_s3
   ```

3. **Descargar el modelo (si no está en el repo):**

   ```bash
   mkdir -p model
   wget -O model/best_model.h5 https://raw.githubusercontent.com/jessicagu039/clasificacion-emociones-audio/main/modelos/best_model.h5
   ```

4. **Ejecutar con Docker Compose:**

   ```bash
   docker-compose up -d
   ```

5. **Verificar que esté funcionando:**

   ```bash
   docker-compose logs -f
   ```

   Accede a: `http://localhost:8001`

# 4. Verificación del Sistema

Para verificar que la instalación fue exitosa, abre tu navegador y visita <http://localhost:8001> (local) o http://IP_PUBLICA:8001 (AWS). Deberías ver la interfaz principal del dashboard. También puedes verificar el estado interno de la aplicación consultando los registros del contenedor ejecutando sudo docker logs api_ser, lo cual confirmará si el modelo y los datos de S3 se inicializaron correctamente en el servidor.
