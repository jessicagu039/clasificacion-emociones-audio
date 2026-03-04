"""
Emotion Classification CNN Pipeline 
--------------------------------------------------------
"""

import os
import io
import logging
import warnings
from typing import List, Tuple, Optional
from collections import Counter

import boto3
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    classification_report, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE AWS S3 Y MLFLOW (CONSTANTES GLOBALES)
# ============================================================================
# Boto3 toma las credenciales del Rol IAM de EC2 o de las variables de entorno.
S3_BUCKET = "amzn-s3-maia-mesd-2026"

# IMPORTANTE: Boto3 requiere el 'Key Prefix' interno, NO la URL HTTPS completa.
S3_PREFIX = "raw/all-wavs/MexicanEmotionalSpeechDatabase/" 

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Emotion_CNN_Model"

# Inicialización de clientes
try:
    s3_client = boto3.client('s3')
except Exception as e:
    logger.error(f"Failed to initialize Boto3 S3 client: {e}")
    raise

# ============================================================================
# FUNCIONES DE INGESTA DE DATOS (AWS S3)
# ============================================================================

def load_audio_from_s3(bucket: str, key: str, sr: int = 16000) -> np.ndarray:
    """
    Downloads an audio file from S3 into memory and converts it to a numpy array.
    
    Args:
        bucket (str): AWS S3 bucket name.
        key (str): The specific file path (key) inside the bucket.
        sr (int): Target sample rate. Defaults to 16000.
        
    Returns:
        np.ndarray: Mono-channel audio time series.
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    audio_bytes = response['Body'].read()
    
    audio, orig_sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    
    # Downmix to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    # Resample if needed
    if sr is not None and orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        
    return audio

def list_all_wav_files(bucket: str, prefix: str = "") -> List[str]:
    """
    Paginates through an S3 bucket prefix to retrieve all .wav file keys.
    """
    wav_keys = []
    continuation_token = None
    
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
            
        response = s3_client.list_objects_v2(**kwargs)

        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.wav'):
                    wav_keys.append(obj['Key'])

        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break
            
    return wav_keys

# ============================================================================
# INGENIERÍA DE CARACTERÍSTICAS & DATA AUGMENTATION
# ============================================================================

def augment_audio(audio: np.ndarray, sr: int = 16000) -> List[np.ndarray]:
    """
    Applies data augmentation strategies to increase dataset variability.
    Generates 5 variations: Original, Pitch Shift (+2/-2), Time Stretch (1.1/0.9).
    """
    return [
        audio,
        librosa.effects.pitch_shift(audio, sr=sr, n_steps=2),
        librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2),
        librosa.effects.time_stretch(audio, rate=1.1),
        librosa.effects.time_stretch(audio, rate=0.9)
    ]

def extract_mel_spectrogram(y: np.ndarray, sr: int = 16000, fixed_duration: float = 2.5) -> Optional[np.ndarray]:
    """
    Transforms an audio signal into a fixed-size 2D Mel-Spectrogram image.
    """
    try:
        # Trim leading and trailing silences
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Standardize length via zero-padding or truncation
        target_length = int(fixed_duration * sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            y = y[:target_length]

        # Peak amplitude normalization
        max_amp = np.max(np.abs(y))
        if max_amp > 0:
            y = y / max_amp

        # Mel-Spectrogram extraction
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512, n_fft=2048)
        S_DB = librosa.power_to_db(S, ref=np.max)

        return S_DB 
    except Exception as e:
        logger.error(f"Spectrogram extraction failed: {e}")
        return None

# ============================================================================
# PREPARACIÓN DEL DATASET
# ============================================================================

def key_to_label(key: str) -> str:
    """Parses the emotion label from the file naming convention."""
    filename = key.split("/")[-1]
    return filename.split("_")[0]

def get_and_split_keys(bucket: str, prefix: str, test_size: float = 0.15, random_state: int = 42) -> Tuple[List[str], List[str], LabelEncoder, int]:
    """
    Orchestrates the S3 listing and stratified train/validation split.
    """
    logger.info("Initiating S3 object listing...")
    keys = list_all_wav_files(bucket, prefix)
    
    if not keys:
        raise ValueError(f"No .wav files found in s3://{bucket}/{prefix}")

    # Shuffling to avoid ordering bias before splitting
    np.random.seed(random_state)
    np.random.shuffle(keys)

    labels = [key_to_label(k) for k in keys]
    
    logger.info(f"Total validated audio files: {len(keys)}")
    logger.info(f"Class distribution: {dict(Counter(labels))}")

    # Encode categorical labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)

    # Stratified split ensures balanced classes in Train and Val
    train_keys, val_keys, _, _ = train_test_split(
        keys, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state
    )
    
    logger.info(f"Data partitioning complete -> Train: {len(train_keys)} | Validation: {len(val_keys)}")
    return train_keys, val_keys, le, num_classes

def build_xy_dataset(bucket: str, keys: List[str], use_aug: bool = False, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downloads, augments, and extracts features for a given list of S3 keys.
    """
    X_list, y_list = [], []
    total = len(keys)
    
    for i, key in enumerate(keys, 1):
        try:
            audio = load_audio_from_s3(bucket, key, sr=sr)
            emotion = key_to_label(key)

            # Apply dynamic augmentation ONLY if requested (prevents data leakage)
            audios_to_process = augment_audio(audio, sr=sr) if use_aug else [audio]
            
            for a in audios_to_process:
                feat = extract_mel_spectrogram(a, sr=sr)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(emotion)

            if i % 50 == 0 or i == total:
                logger.info(f"Processing progress: {i}/{total} files")
                
        except Exception as e:
            logger.warning(f"Failed to process {key}: {e}")
            continue

    return np.array(X_list), np.array(y_list)

# ============================================================================
# EVALUACIÓN Y MLFLOW ARTIFACTS
# ============================================================================

def log_metrics_to_mlflow(y_train: np.ndarray, y_pred_train: np.ndarray, 
                          y_val: np.ndarray, y_pred_val: np.ndarray, 
                          le: LabelEncoder, model_tag: str = "model") -> Tuple[float, float]:
    """
    Calculates classification metrics and logs them to the active MLflow run,
    including confusion matrix and classification report artifacts.
    """
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc   = accuracy_score(y_val, y_pred_val)
    train_f1  = f1_score(y_train, y_pred_train, average='macro')
    val_f1    = f1_score(y_val, y_pred_val, average='macro')

    mlflow.log_metrics({
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "train_f1_macro": float(train_f1),
        "val_f1_macro": float(val_f1),
        "gap_f1": float(train_f1 - val_f1)
    })

    # Detailed per-class metrics
    prec, rec, f1c, _ = precision_recall_fscore_support(
        y_val, y_pred_val, labels=np.unique(y_val), zero_division=0
    )
    for i, cls_name in enumerate(le.classes_):
        mlflow.log_metrics({
            f"val_precision_{cls_name}": float(prec[i]),
            f"val_recall_{cls_name}": float(rec[i]),
            f"val_f1_{cls_name}": float(f1c[i])
        })

    # Safe handling of local temporary files for MLflow artifacts
    report_path = f"{model_tag}_classification_report.txt"
    cm_path = f"{model_tag}_confusion_matrix.png"
    
    try:
        report = classification_report(y_val, y_pred_val, target_names=le.classes_, digits=4, zero_division=0)
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        cm = confusion_matrix(y_val, y_pred_val)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        fig, ax = plt.subplots(figsize=(8,6))
        disp.plot(ax=ax, xticks_rotation=45, cmap='Blues')
        plt.title('Confusion Matrix - CNN')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(cm_path)
        
    finally:
        # Guarantee cleanup of VM disk space
        if os.path.exists(report_path): os.remove(report_path)
        if os.path.exists(cm_path): os.remove(cm_path)

    return train_f1, val_f1

# ============================================================================
# ARQUITECTURA DE RED NEURONAL (CNN)
# ============================================================================

def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Constructs a lightweight, VGG-inspired 2D Convolutional Neural Network.
    Designed with aggressive Dropout and BatchNormalization to prevent overfitting
    on small acoustic datasets.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.1),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.15),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Dense Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================================
# PIPELINE PRINCIPAL (ENTRY POINT)
# ============================================================================

def execute_pipeline():
    """Main orchestrator for the Deep Learning training pipeline."""
    logger.info("=== STARTING EMOTION CNN PIPELINE ===")
    
    # 1. Configure MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 2. Retrieve and split dataset keys
    train_keys, val_keys, le, num_classes = get_and_split_keys(S3_BUCKET, S3_PREFIX)

    # 3. Distributed Feature Extraction
    logger.info("Extracting TRAIN dataset features (Applying Data Augmentation x5)...")
    X_train, y_train_str = build_xy_dataset(S3_BUCKET, train_keys, use_aug=True)

    logger.info("Extracting VALIDATION dataset features (Clean, no augmentation)...")
    X_val, y_val_str = build_xy_dataset(S3_BUCKET, val_keys, use_aug=False)

    # 4. Label Encoding
    y_train = le.transform(y_train_str)
    y_val   = le.transform(y_val_str)

    # Add channel dimension for CNN compatibility: (Samples, Height, Width, Channels)
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)

    logger.info(f"Final Tensor Shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}")

    # 5. MLflow Tracking Context
    with mlflow.start_run(run_name="CNN_MelSpectrogram_Model") as run:
        
        # Log Hyperparameters
        mlflow.log_params({
            "model_type": "CNN_2D_Light_Dropout",
            "feature_extraction": "Mel_Spectrogram",
            "n_mels": 128,
            "fixed_duration": 2.5,
            "augmentation": "train_only_5x",
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.0003
        })

        # 6. Model Compilation
        input_shape = X_train.shape[1:] 
        model = build_cnn_model(input_shape, num_classes)
        
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True, 
            verbose=1
        )

        # 7. Model Training
        logger.info("Commencing CNN Model Training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

        # 8. Learning Curves Visualization
        logger.info("Generating learning curves...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#4A90E2')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#FF9F1C')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.plot(history.history['loss'], label='Train Loss', color='#4A90E2')
        ax2.plot(history.history['val_loss'], label='Validation Loss', color='#FF9F1C')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

        history_plot_path = "learning_curves.png"
        try:
            plt.tight_layout()
            plt.savefig(history_plot_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(history_plot_path)
        finally:
            if os.path.exists(history_plot_path): os.remove(history_plot_path)

        # 9. Evaluation & Artifact Logging
        logger.info("Evaluating model and computing metrics...")
        y_pred_probs_train = model.predict(X_train)
        y_pred_probs_val = model.predict(X_val)
        
        y_pred_train = np.argmax(y_pred_probs_train, axis=1)
        y_pred_val = np.argmax(y_pred_probs_val, axis=1)

        train_f1, val_f1 = log_metrics_to_mlflow(
            y_train, y_pred_train, y_val, y_pred_val, le, model_tag="CNN_Audio"
        )
        
        mlflow.tensorflow.log_model(model, "model")

        # 10. Final Summary
        logger.info("=== CNN TRAINING COMPLETED SUCESSFULLY ===")
        logger.info(f"Train F1 (Macro): {train_f1:.4f}")
        logger.info(f"Val F1 (Macro):   {val_f1:.4f}")
        logger.info(f"Overfitting Gap:  {(train_f1 - val_f1):.4f}")
        logger.info(f"MLflow Run ID:    {run.info.run_id}")
        logger.info("==========================================")

if __name__ == "__main__":
    execute_pipeline()