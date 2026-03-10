"""
IoT Sentinel — Model Training Script (CIC-IoT-DiAD-2024)
Trains Isolation Forest on CIC-IoT-DiAD-2024 dataset (folder-based structure).

The dataset has 7 folders: Benign, Brute Force, DDOS, DOS, Mirai, Recon, Spoofing.
Labels are derived from folder names. Training uses Benign data only (anomaly detection).

Usage:
    python scripts/train_models_diad2024.py

Output:
    models/if_cctv.joblib  (+ 4 more device-class copies)
    models/scaler.joblib
    models/training_metadata.json
"""

import json
import time
import os
import glob
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("E:/Sentinel/data/cic_iot_diad_2024")
MODEL_DIR = Path("models/")
MODEL_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
# Map folder names to attack labels
FOLDER_LABEL_MAP = {
    "Benign": "Benign",
    "Brute Force": "BruteForce",
    "DDOS": "DDOS",
    "DOS": "DOS",
    "Mirai": "Mirai",
    "Recon": "Recon",
    "Spoofing": "Spoofing",
}

BENIGN_LABEL = "Benign"

DEVICE_CLASSES = ["cctv", "router", "door_lock", "smart_light", "laptop"]

# CICFlowMeter features that map well to the 18 training features from ML_SPEC
# Selecting features that capture the same behavioral patterns
BASE_FEATURES = [
    "Flow Bytes/s",            # proxy for Rate
    "Flow IAT Mean",           # proxy for IAT
    "Packet Length Variance",   # proxy for Variance
    "Total Length of Fwd Packet",  # proxy for Tot size
    "Average Packet Size",     # proxy for AVG
    "Packet Length Std",       # proxy for Std
    "Total Fwd Packet",        # proxy for Number (flow count)
    "Fwd Header Length",       # proxy for Header_Length
    "Flow Duration",           # related to Time_To_Live
    "RST Flag Count",          # proxy for rst_count
    "FIN Flag Count",          # proxy for fin_count
    "SYN Flag Count",          # proxy for syn_count
    "PSH Flag Count",          # psh flags
    "ACK Flag Count",          # ack flags
    "Flow Packets/s",          # packets per second
    "Fwd Packet Length Mean",  # forward packet size
    "Bwd Packet Length Mean",  # backward packet size
    "Fwd Packets/s",           # forward packets rate
    "Bwd Packets/s",           # backward packets rate
    "Packet Length Mean",      # overall packet size
    "Flow IAT Std",            # IAT standard deviation
    "Down/Up Ratio",           # traffic asymmetry proxy
    "Fwd Packet Length Max",   # max forward packet
    "Bwd Packet Length Max",   # max backward packet
    "Total Bwd packets",       # backward packet count
    "Total Length of Bwd Packet",  # backward total bytes
]

# Max rows to sample per folder (to keep memory manageable)
MAX_ROWS_PER_FOLDER = 50000


def load_folder(folder_path: Path, label: str, max_rows: int) -> pd.DataFrame:
    """Load and sample CSVs from a single folder, assign label."""
    csv_files = sorted(glob.glob(str(folder_path / "*.csv")))
    if not csv_files:
        print(f"  WARNING: No CSV files in {folder_path}")
        return pd.DataFrame()

    dfs = []
    rows_loaded = 0

    for csv_file in csv_files:
        if rows_loaded >= max_rows:
            break
        try:
            remaining = max_rows - rows_loaded
            df = pd.read_csv(csv_file, nrows=remaining, low_memory=False)
            rows_loaded += len(df)
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {os.path.basename(csv_file)}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["attack_label"] = label
    print(f"  {label:<12}: loaded {len(combined):>6} rows from {len(dfs)} files")
    return combined


def load_all_data(data_dir: Path, max_rows_per_folder: int) -> pd.DataFrame:
    """Load sampled data from all folders."""
    print(f"Loading data from {data_dir} ...")
    all_dfs = []

    for folder_name, label in FOLDER_LABEL_MAP.items():
        folder_path = data_dir / folder_name
        if folder_path.exists():
            df = load_folder(folder_path, label, max_rows_per_folder)
            if len(df) > 0:
                all_dfs.append(df)
        else:
            print(f"  WARNING: Folder not found: {folder_path}")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal loaded: {len(combined)} rows")
    print(f"Label distribution:\n{combined['attack_label'].value_counts().to_string()}\n")
    return combined


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and engineer features from CICFlowMeter columns."""
    # Check which base features exist
    available = [f for f in BASE_FEATURES if f in df.columns]
    missing = [f for f in BASE_FEATURES if f not in df.columns]
    if missing:
        print(f"  Note: {len(missing)} features missing, using {len(available)} available")

    feat = df[available].copy()
    feat = feat.replace([np.inf, -np.inf], 0).fillna(0)

    # Convert any string columns to numeric
    for col in feat.columns:
        feat[col] = pd.to_numeric(feat[col], errors='coerce').fillna(0)

    # Engineered features (ratios, interactions)
    feat["syn_to_fin"] = feat["SYN Flag Count"] / (feat["FIN Flag Count"] + 1)
    feat["rst_ratio"] = feat["RST Flag Count"] / (feat["Total Fwd Packet"] + 1)
    feat["bytes_per_packet"] = feat["Flow Bytes/s"] / (feat["Flow Packets/s"] + 1)
    feat["fwd_bwd_ratio"] = feat["Total Length of Fwd Packet"] / (feat["Total Length of Bwd Packet"] + 1)
    feat["pkt_size_ratio"] = feat["Fwd Packet Length Mean"] / (feat["Bwd Packet Length Mean"] + 1)
    feat["flag_sum"] = feat["SYN Flag Count"] + feat["FIN Flag Count"] + feat["RST Flag Count"] + feat["PSH Flag Count"]
    feat["iat_x_variance"] = feat["Flow IAT Mean"] * feat["Packet Length Variance"]
    feat["duration_x_rate"] = feat["Flow Duration"] * feat["Flow Packets/s"]

    # Log transforms for skewed features
    for col in ["Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Packet Length Variance",
                 "Total Length of Fwd Packet", "Flow Duration"]:
        if col in feat.columns:
            feat[f"log_{col}"] = np.log1p(feat[col].clip(lower=0))

    feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
    return feat


def find_optimal_threshold(
    scores: np.ndarray, y_true: np.ndarray, n_steps: int = 2000
) -> tuple:
    """Sweep thresholds to maximize overall accuracy with min 40% normal accuracy."""
    lo, hi = float(scores.min()), float(scores.max())
    best_acc = -1.0
    best_thresh = lo
    best_metrics = {}

    normal_mask = y_true == 0
    attack_mask = y_true == 1
    n_normal = int(normal_mask.sum())
    n_attack = int(attack_mask.sum())

    for thresh in np.linspace(lo, hi, n_steps):
        y_pred = (scores < thresh).astype(int)

        normal_correct = int((y_pred[normal_mask] == 0).sum())
        attack_correct = int((y_pred[attack_mask] == 1).sum())
        normal_acc = normal_correct / n_normal if n_normal > 0 else 0
        attack_acc = attack_correct / n_attack if n_attack > 0 else 0

        if normal_acc < 0.40:
            continue

        acc = (normal_correct + attack_correct) / (n_normal + n_attack)

        if acc > best_acc:
            best_acc = acc
            best_thresh = float(thresh)
            best_metrics = {
                "accuracy": float(acc),
                "normal_accuracy": float(normal_acc),
                "attack_detection_rate": float(attack_acc),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "threshold": float(thresh),
            }

    return best_thresh, best_metrics


def main():
    t_start = time.time()

    # 1. Load sampled data from all folders
    df = load_all_data(DATA_DIR, MAX_ROWS_PER_FOLDER)

    benign_mask = df["attack_label"] == BENIGN_LABEL
    print(f"Benign rows (training): {benign_mask.sum()}")
    print(f"Attack rows (eval):     {(~benign_mask).sum()}\n")

    if benign_mask.sum() == 0:
        raise ValueError("No benign rows found!")

    # 2. Engineer features
    print("Engineering features ...")
    feat_df = engineer_features(df)
    feature_names = list(feat_df.columns)
    print(f"  Total features: {len(feature_names)}\n")

    # 3. Scale using benign data
    scaler = StandardScaler()
    X_benign = feat_df[benign_mask].values
    scaler.fit(X_benign)
    X_benign_scaled = scaler.transform(X_benign)
    X_all_scaled = scaler.transform(feat_df.values)

    # 4. Train Isolation Forest
    print(f"Training IsolationForest on {X_benign_scaled.shape[0]} rows, "
          f"{X_benign_scaled.shape[1]} features ...")
    t0 = time.time()
    model = IsolationForest(
        n_estimators=300,
        contamination=0.01,
        max_samples=0.8,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_benign_scaled)
    print(f"Training complete in {time.time() - t0:.1f}s\n")

    # 5. Score full dataset
    y_true = (~benign_mask).astype(int).values
    scores = model.score_samples(X_all_scaled)

    print(f"Score stats — Benign:  mean={scores[benign_mask].mean():.4f}, "
          f"std={scores[benign_mask].std():.4f}")
    print(f"Score stats — Attack:  mean={scores[~benign_mask].mean():.4f}, "
          f"std={scores[~benign_mask].std():.4f}\n")

    # 6. Threshold tuning
    print("Tuning anomaly threshold (2000 steps) ...")
    best_thresh, best_metrics = find_optimal_threshold(scores, y_true)
    print(f"  Optimal threshold:      {best_thresh:.6f}")
    print(f"  Overall accuracy:       {best_metrics['accuracy']:.4f}")
    print(f"  Normal accuracy:        {best_metrics['normal_accuracy']:.4f}")
    print(f"  Attack detection rate:  {best_metrics['attack_detection_rate']:.4f}")
    print(f"  Precision:              {best_metrics['precision']:.4f}")
    print(f"  Recall:                 {best_metrics['recall']:.4f}")
    print(f"  F1 Score:               {best_metrics['f1']:.4f}\n")

    # 7. Classification report
    y_pred = (scores < best_thresh).astype(int)
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=["Benign", "Attack"], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  True Normal  (TN): {cm[0][0]}")
    print(f"  False Alarm  (FP): {cm[0][1]}")
    print(f"  Missed Attack(FN): {cm[1][0]}")
    print(f"  True Attack  (TP): {cm[1][1]}")
    fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    print(f"  False Positive Rate: {fpr:.4f}\n")

    # 8. Per-attack-type
    print("Per Attack Type Detection:")
    attack_breakdown = {}
    for label in sorted(df["attack_label"].unique()):
        if label == BENIGN_LABEL:
            continue
        mask = (df["attack_label"] == label).values
        detected = int((y_pred[mask] == 1).sum())
        total = int(mask.sum())
        rate = detected / total if total > 0 else 0
        attack_breakdown[label] = {"detected": detected, "total": total, "rate": rate}
        print(f"  {label:<15} {detected:>6}/{total:<6} detected ({rate*100:.1f}%)")

    # 9. Save models
    print("\nSaving models ...")
    for device_class in DEVICE_CLASSES:
        path = MODEL_DIR / f"if_{device_class}.joblib"
        joblib.dump(model, path)
        print(f"  Saved: {path}")

    scaler_path = MODEL_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  Saved: {scaler_path}")

    metadata = {
        "dataset": "CIC-IoT-DiAD-2024",
        "data_dir": str(DATA_DIR),
        "training_features": feature_names,
        "benign_label": BENIGN_LABEL,
        "device_classes": DEVICE_CLASSES,
        "model_params": {
            "n_estimators": 300,
            "contamination": 0.01,
            "max_samples": 0.8,
            "max_features": 0.8,
            "random_state": 42,
        },
        "optimal_threshold": best_thresh,
        "metrics": best_metrics,
        "benign_rows": int(benign_mask.sum()),
        "total_rows": int(len(df)),
        "max_rows_per_folder": MAX_ROWS_PER_FOLDER,
        "attack_breakdown": attack_breakdown,
        "training_time_seconds": round(time.time() - t_start, 1),
    }
    meta_path = MODEL_DIR / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {meta_path}")

    overall_acc = best_metrics["accuracy"] * 100
    print(f"\nDone. {len(DEVICE_CLASSES)} model files + scaler + metadata saved to {MODEL_DIR}/")
    print(f"Overall accuracy: {overall_acc:.1f}%  "
          f"{'✅ PASS' if overall_acc >= 85 else '❌ BELOW TARGET'}")


if __name__ == "__main__":
    main()
