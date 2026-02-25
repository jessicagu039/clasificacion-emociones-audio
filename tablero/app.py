import os
import base64
import io
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend no interactivo para servidores
import matplotlib.pyplot as plt
import librosa
import librosa.display
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

# --- CARGA DEL MODELO Y CLASES ---
MODEL_PATH = 'model/cnn_emotion_model.h5'
JSON_PATH = 'model/class_mapping.json'

# Cargar el modelo de Keras
modelo_emociones = load_model(MODEL_PATH)

# Cargar el mapeo de clases dinámicamente
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    EMOCIONES_MAP = json.load(f)
# --- FIN DE LA CARGA ---

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # 1. Guardar archivo temporalmente
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # 2. Generar Espectrograma para la UI (Visualización)
            spectrogram_b64 = generate_spectrogram(filepath)

            # 3. Realizar Inferencia REAL con Keras CNN
            result = real_prediction(filepath) 

            # 4. Limpieza (Opcional)
            # os.remove(filepath)

            return jsonify({
                'success': True,
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'spectrogram_image': spectrogram_b64
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

def generate_spectrogram(audio_path):
    """
    Carga el audio, crea un espectrograma y lo devuelve como string base64
    para mostrarlo en HTML sin guardar un archivo de imagen.
    """
    y, sr = librosa.load(audio_path)
    
    plt.figure(figsize=(10, 4))
    # Generar espectrograma de Mel para la visualización (UI)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

    # Guardar en buffer de memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    
    # Convertir a base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close() # Cerrar figura para liberar memoria
    
    return f"data:image/png;base64,{image_base64}"

def preprocesar_audio(audio_path, sr=16000, fixed_duration=2.5):
    """
    Convierte el audio en un Mel-Espectrograma idéntico al del entrenamiento.
    """
    y, _ = librosa.load(audio_path, sr=sr)
    
    # 1. Recortar silencios en los extremos
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # 2. Padding o truncamiento a longitud fija (2.5 segundos)
    target_length = int(fixed_duration * sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]

    # 3. Normalización de volumen
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # 4. Generar Mel-Espectrograma con tus parámetros exactos
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512, n_fft=2048)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # 5. Ajustar dimensiones para la CNN (Batch_size, Height, Width, Channels)
    features = np.expand_dims(S_DB, axis=0) # Añade dimensión de batch (1, 128, X)
    features = np.expand_dims(features, axis=-1) # Añade dimensión de canal (1, 128, X, 1)

    return features

def real_prediction(filepath):
    """
    Ejecuta el audio real a través del modelo Keras cargado.
    """
    # 1. Extraer características (la matriz 4D)
    features = preprocesar_audio(filepath)
    
    # 2. Inferencia con la CNN
    predicciones = modelo_emociones.predict(features)
    
    # 3. Extraer probabilidades y pasarlas a porcentajes
    probs = predicciones[0] * 100 
    probs_redondeadas = [round(float(p), 1) for p in probs]
    
    # 4. Obtener la clase ganadora
    max_idx = np.argmax(probs_redondeadas)
    
    # 5. Mapear el índice al nombre real de la emoción usando tu JSON
    emocion_detectada = EMOCIONES_MAP[str(max_idx)]
    confianza = probs_redondeadas[max_idx]
    
    return {
        'emotion': emocion_detectada,
        'confidence': confianza,
        'probabilities': probs_redondeadas
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
