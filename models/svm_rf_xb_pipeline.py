
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    classification_report, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier

# ============================================================================
# CONFIGURACIÓN DE LOGGING Y WARNINGS
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE AWS S3 Y MLFLOW (
# ============================================================================
S3_BUCKET = "amzn-s3-maia-mesd-2026"
S3_PREFIX = "raw/all-wavs/MexicanEmotionalSpeechDatabase/" 

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Emotion_Classical_Models"

# Inicialización del cliente S3
try:
    s3_client = boto3.client('s3')
except Exception as e:
    logger.error(f"Failed to initialize Boto3 S3 client: {e}")
    raise

# ============================================================================
# FUNCIONES DE INGESTA DE DATOS (AWS S3)
# ============================================================================

def load_audio_from_s3(bucket: str, key: str, sr: int = 16000) -> np.ndarray:
    """Downloads and decodes an audio file from S3 into a numpy array."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    audio_bytes = response['Body'].read()
    
    audio, orig_sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    if sr is not None and orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        
    return audio

def list_all_wav_files(bucket: str, prefix: str = "") -> List[str]:
    """Paginates through an S3 bucket to retrieve all .wav file keys."""
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
    """Applies pitch shifting and time stretching to increase dataset variability."""
    return [
        audio,
        librosa.effects.pitch_shift(audio, sr=sr, n_steps=2),
        librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2),
        librosa.effects.time_stretch(audio, rate=1.1),
        librosa.effects.time_stretch(audio, rate=0.9)
    ]

def extract_features_audio(y: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
    """
    Extracts a robust 288-dimensional handcrafted feature vector:
    MFCCs, Chroma, Spectral (Centroid, Bandwidth, Rolloff), ZCR, RMS, and Pitch.
    """
    try:
        y, _ = librosa.effects.trim(y, top_db=20)

        MIN_DURATION = 1.0
        if len(y) < int(MIN_DURATION * sr):
            y = np.pad(y, (0, int(MIN_DURATION * sr) - len(y)), mode="constant")

        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        features = []
        HOP_LENGTH = 512
        N_FFT = 2048

        # 1. MFCCs and Deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        for M in [mfcc, mfcc_delta, mfcc_delta2]:
            features.extend(np.mean(M, axis=1))
            features.extend(np.std(M, axis=1))
            features.extend(np.min(M, axis=1))
            features.extend(np.max(M, axis=1))

        # 2. Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # 3. Spectral Features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)

        for feat in [centroid, bandwidth, rolloff]:
            features.extend([float(np.mean(feat)), float(np.std(feat)), float(np.min(feat)), float(np.max(feat))])

        # 4. Zero Crossing Rate & 5. RMS
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        
        for feat in [zcr, rms]:
            features.extend([float(np.mean(feat)), float(np.std(feat)), float(np.min(feat)), float(np.max(feat))])

        # 6. Fundamental Frequency (Pitch)
        try:
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                features.extend([float(np.mean(f0_clean)), float(np.std(f0_clean)), 
                                 float(np.min(f0_clean)), float(np.max(f0_clean))])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        except Exception:
            features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Feature extraction failed: {e}")
        return None

# ============================================================================
# PREPARACIÓN DEL DATASET
# ============================================================================

def key_to_label(key: str) -> str:
    """Parses the emotion label from the file naming convention."""
    filename = key.split("/")[-1]
    return filename.split("_")[0]

def get_and_split_keys(bucket: str, prefix: str, test_size: float = 0.15, random_state: int = 42) -> Tuple[List[str], List[str], LabelEncoder]:
    """Orchestrates the S3 listing and stratified train/validation split."""
    logger.info("Initiating S3 object listing...")
    keys = list_all_wav_files(bucket, prefix)
    
    if not keys:
        raise ValueError(f"No .wav files found in s3://{bucket}/{prefix}")

    np.random.seed(random_state)
    np.random.shuffle(keys)

    labels = [key_to_label(k) for k in keys]
    logger.info(f"Total validated audio files: {len(keys)}")
    logger.info(f"Class distribution: {dict(Counter(labels))}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    train_keys, val_keys, _, _ = train_test_split(
        keys, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state
    )
    
    logger.info(f"Data partitioning complete -> Train: {len(train_keys)} | Validation: {len(val_keys)}")
    return train_keys, val_keys, le

def build_xy_dataset(bucket: str, keys: List[str], use_aug: bool = False, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """Downloads, augments, and extracts 1D features for a list of S3 keys."""
    X_list, y_list = [], []
    total = len(keys)
    
    for i, key in enumerate(keys, 1):
        try:
            audio = load_audio_from_s3(bucket, key, sr=sr)
            emotion = key_to_label(key)

            audios_to_process = augment_audio(audio, sr=sr) if use_aug else [audio]
            
            for a in audios_to_process:
                feat = extract_features_audio(a, sr=sr)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(emotion)

            if i % 50 == 0 or i == total:
                logger.info(f"Feature extraction progress: {i}/{total} files")
                
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
    """Calculates classification metrics and logs them to the active MLflow run."""
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

    prec, rec, f1c, _ = precision_recall_fscore_support(
        y_val, y_pred_val, labels=np.unique(y_val), zero_division=0
    )
    for i, cls_name in enumerate(le.classes_):
        mlflow.log_metrics({
            f"val_precision_{cls_name}": float(prec[i]),
            f"val_recall_{cls_name}": float(rec[i]),
            f"val_f1_{cls_name}": float(f1c[i])
        })

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
        plt.title(f'Confusion Matrix - {model_tag}')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(cm_path)
        
    finally:
        if os.path.exists(report_path): os.remove(report_path)
        if os.path.exists(cm_path): os.remove(cm_path)

    return train_f1, val_f1

# ============================================================================
# PIPELINE PRINCIPAL (ENTRY POINT)
# ============================================================================

def execute_pipeline():
    logger.info("=== STARTING CLASSICAL ML PIPELINE ===")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    train_keys, val_keys, le = get_and_split_keys(S3_BUCKET, S3_PREFIX)

    logger.info("Extracting TRAIN features (Applying Data Augmentation x5)...")
    X_train, y_train_str = build_xy_dataset(S3_BUCKET, train_keys, use_aug=True)

    logger.info("Extracting VALIDATION features (Clean, no augmentation)...")
    X_val, y_val_str = build_xy_dataset(S3_BUCKET, val_keys, use_aug=False)

    y_train = le.transform(y_train_str)
    y_val   = le.transform(y_val_str)

    logger.info(f"Final Feature Matrix -> X_train: {X_train.shape}, X_val: {X_val.shape}")

    with mlflow.start_run(run_name="Classical_Models_Suite") as parent_run:
        
        mlflow.log_params({
            "split_strategy": "by_file",
            "augmentation": "train_only_5x",
            "train_samples_augmented": int(len(X_train)),
            "val_samples_original": int(len(X_val)),
        })

        # ====================================================================
        # MODEL 1: SVM (Baseline)
        # ====================================================================
        with mlflow.start_run(run_name="SVM_RBF", nested=True) as run:
            logger.info("Training Model 1: Support Vector Machine (RBF)")
            
            svm_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.80, random_state=42)),
                ('svm', SVC(C=0.676, gamma=0.00240, kernel='rbf', class_weight='balanced', random_state=42))
            ])
            
            mlflow.log_params({"model": "SVC", "pca_variance": 0.80, "C": 0.676, "gamma": 0.00240})
            
            svm_pipeline.fit(X_train, y_train)
            t_f1, v_f1 = log_metrics_to_mlflow(
                y_train, svm_pipeline.predict(X_train), y_val, svm_pipeline.predict(X_val), le, "SVM_RBF"
            )
            mlflow.sklearn.log_model(svm_pipeline, "model")
            logger.info(f"SVM Completed -> Train F1: {t_f1:.4f} | Val F1: {v_f1:.4f}")

        # ====================================================================
        # MODEL 2: Random Forest (Highly Regularized)
        # ====================================================================
        with mlflow.start_run(run_name="RF_Highly_Regularized", nested=True) as run:
            logger.info("Training Model 2: Random Forest (No PCA)")
            
            rf_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(
                    n_estimators=500, max_depth=6, min_samples_split=60, 
                    min_samples_leaf=20, max_features=0.1, max_samples=0.6,
                    bootstrap=True, class_weight='balanced', random_state=42, n_jobs=-1
                ))
            ])
            
            mlflow.log_params({"model": "RandomForest", "pca": False, "max_depth": 6, "min_samples_split": 60})
            
            rf_pipeline.fit(X_train, y_train)
            t_f1, v_f1 = log_metrics_to_mlflow(
                y_train, rf_pipeline.predict(X_train), y_val, rf_pipeline.predict(X_val), le, "RF"
            )
            mlflow.sklearn.log_model(rf_pipeline, "model")
            logger.info(f"RF Completed -> Train F1: {t_f1:.4f} | Val F1: {v_f1:.4f}")

        # ====================================================================
        # MODEL 3: XGBoost (Highly Regularized)
        # ====================================================================
        with mlflow.start_run(run_name="XGB_Highly_Regularized", nested=True) as run:
            logger.info("Training Model 3: XGBoost")
            
            xgb_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.75, random_state=42)),
                ('xgb', XGBClassifier(
                    n_estimators=200, learning_rate=0.03, max_depth=3, min_child_weight=20,
                    subsample=0.6, colsample_bytree=0.5, gamma=3.0, reg_alpha=2.0, reg_lambda=10.0,
                    objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=-1
                ))
            ])
            
            mlflow.log_params({"model": "XGBoost", "pca_variance": 0.75, "learning_rate": 0.03, "max_depth": 3})
            
            xgb_pipeline.fit(X_train, y_train)
            t_f1, v_f1 = log_metrics_to_mlflow(
                y_train, xgb_pipeline.predict(X_train), y_val, xgb_pipeline.predict(X_val), le, "XGBoost"
            )
            mlflow.sklearn.log_model(xgb_pipeline, "model")
            logger.info(f"XGBoost Completed -> Train F1: {t_f1:.4f} | Val F1: {v_f1:.4f}")

    logger.info("=== ALL CLASSICAL MODELS COMPLETED SUCESSFULLY ===")

if __name__ == "__main__":

    execute_pipeline()
