import os
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend no interactivo para servidores
import matplotlib.pyplot as plt
import librosa
import librosa.display
from flask import Flask, render_template, request, jsonify

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
            # 2. Generar Espectrograma (Imagen en base64)
            spectrogram_b64 = generate_spectrogram(filepath)

            # 3. Realizar Inferencia (Aquí llamas a tu modelo)
            # result = mi_modelo.predict(filepath) <--- TU LÓGICA AQUÍ
            result = dummy_prediction(filepath) 

            # 4. Limpieza (Opcional: borrar archivo después de procesar)
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
    # Generar espectrograma de Mel
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

def dummy_prediction(filepath):
    """
    SIMULACIÓN DEL MODELO.
    Reemplaza esto con: load_model() y model.predict()
    """
    # Lógica Dummy
    emotions = ['Alegría', 'Tristeza', 'Enojo', 'Miedo', 'Neutral']
    probs = np.random.dirichlet(np.ones(5), size=1)[0] * 100
    probs = [round(p, 1) for p in probs]
    
    max_idx = np.argmax(probs)
    
    return {
        'emotion': emotions[max_idx],
        'confidence': probs[max_idx],
        'probabilities': probs # Debe coincidir con el orden de emotions
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)