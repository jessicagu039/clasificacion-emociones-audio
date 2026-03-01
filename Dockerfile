FROM python:3.10

# Instalar dependencias del sistema para audio
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Carpeta base del contenedor
WORKDIR /app

# Copiamos el archivo de requerimientos que está dentro de la carpeta tablero
COPY tablero/requirements.txt .
RUN pip install --upgrade pip && \
    pip install "urllib3<2" && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos todo el contenido del repo al contenedor
COPY . .

# IMPORTANTE: Nos movemos a la subcarpeta donde vive realmente tu código
WORKDIR /app/tablero

EXPOSE 5000

# Ahora Gunicorn encontrará 'app.py' porque estamos parados dentro de 'tablero'
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
