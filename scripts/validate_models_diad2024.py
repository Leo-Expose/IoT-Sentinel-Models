"""
IoT Sentinel — Model Validation Script (CIC-IoT-DiAD-2024)
Loads trained model + scaler and validates against freshly loaded data.

Usage:
    python scripts/validate_models_diad2024.py
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_DIR = Path("E:/Sentinel/data/cic_iot_diad_2024")
MODEL_DIR = Path("models/")

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

BASE_FEATURES = [
    "Flow Bytes/s", "Flow IAT Mean", "Packet Length Variance",
    "Total Length of Fwd Packet", "Average Packet Size", "Packet Length Std",
    "Total Fwd Packet", "Fwd Header Length", "Flow Duration",
    "RST Flag Count", "FIN Flag Count", "SYN Flag Count",
    "PSH Flag Count", "ACK Flag Count", "Flow Packets/s",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Fwd Packets/s", "Bwd Packets/s", "Packet Length Mean",
    "Flow IAT Std", "Down/Up Ratio", "Fwd Packet Length Max",
    "Bwd Packet Length Max", "Total Bwd packets", "Total Length of Bwd Packet",
]

# Use different sample for validation — skip first N rows, take next N
VAL_SKIP_ROWS = 50000
VAL_MAX_ROWS = 20000


def load_folder_validation(folder_path: Path, label: str, skip: int, max_rows: int) -> pd.DataFrame:
    """Load validation rows from a folder — skip training rows, take next batch."""
    csv_files = sorted(glob.glob(str(folder_path / "*.csv")))
    if not csv_files:
        return pd.DataFrame()

    dfs = []
    rows_skipped = 0
    rows_loaded = 0

    for csv_file in csv_files:
        if rows_loaded >= max_rows:
            break
        try:
            # Count total rows in file
            df_full = pd.read_csv(csv_file, low_memory=False)
            file_rows = len(df_full)

            if rows_skipped + file_rows <= skip:
                rows_skipped += file_rows
                continue

            # Calculate offset within this file
            offset_in_file = max(0, skip - rows_skipped)
            remaining = max_rows - rows_loaded
            chunk = df_full.iloc[offset_in_file:offset_in_file + remaining]
            rows_skipped += offset_in_file
            rows_loaded += len(chunk)
            if len(chunk) > 0:
                dfs.append(chunk)
        except Exception as e:
            print(f"  WARNING: Could not read {os.path.basename(csv_file)}: {e}")
            continue

    if not dfs:
        # Fall back — if no rows after skip, just take the last available
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, nrows=max_rows, low_memory=False)
                df["attack_label"] = label
                return df
            except:
                continue
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["attack_label"] = label
    return combined


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror feature engineering from training."""
    available = [f for f in BASE_FEATURES if f in df.columns]
    feat = df[available].copy()
    feat = feat.replace([np.inf, -np.inf], 0).fillna(0)

    for col in feat.columns:
        feat[col] = pd.to_numeric(feat[col], errors='coerce').fillna(0)

    feat["syn_to_fin"] = feat["SYN Flag Count"] / (feat["FIN Flag Count"] + 1)
    feat["rst_ratio"] = feat["RST Flag Count"] / (feat["Total Fwd Packet"] + 1)
    feat["bytes_per_packet"] = feat["Flow Bytes/s"] / (feat["Flow Packets/s"] + 1)
    feat["fwd_bwd_ratio"] = feat["Total Length of Fwd Packet"] / (feat["Total Length of Bwd Packet"] + 1)
    feat["pkt_size_ratio"] = feat["Fwd Packet Length Mean"] / (feat["Bwd Packet Length Mean"] + 1)
    feat["flag_sum"] = feat["SYN Flag Count"] + feat["FIN Flag Count"] + feat["RST Flag Count"] + feat["PSH Flag Count"]
    feat["iat_x_variance"] = feat["Flow IAT Mean"] * feat["Packet Length Variance"]
    feat["duration_x_rate"] = feat["Flow Duration"] * feat["Flow Packets/s"]

    for col in ["Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Packet Length Variance",
                 "Total Length of Fwd Packet", "Flow Duration"]:
        if col in feat.columns:
            feat[f"log_{col}"] = np.log1p(feat[col].clip(lower=0))

    feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
    return feat


def main():
    # Load metadata
    meta_path = MODEL_DIR / "training_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    threshold = metadata["optimal_threshold"]
    print(f"Optimal threshold from training: {threshold:.6f}")

    # Load model & scaler
    model = joblib.load(MODEL_DIR / "if_cctv.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    print("Model and scaler loaded.\n")

    # Load validation data (different rows than training)
    print(f"Loading VALIDATION data (skip first {VAL_SKIP_ROWS}, take up to {VAL_MAX_ROWS} per folder)...")
    all_dfs = []
    for folder_name, label in FOLDER_LABEL_MAP.items():
        folder_path = DATA_DIR / folder_name
        if folder_path.exists():
            df = load_folder_validation(folder_path, label, VAL_SKIP_ROWS, VAL_MAX_ROWS)
            if len(df) > 0:
                print(f"  {label:<12}: {len(df):>6} rows")
                all_dfs.append(df)

    val_df = pd.concat(all_dfs, ignore_index=True)
    benign_mask = val_df["attack_label"] == BENIGN_LABEL
    print(f"\nValidation set: {len(val_df)} total, {benign_mask.sum()} benign, {(~benign_mask).sum()} attack\n")

    # Engineer & scale
    feat_df = engineer_features(val_df)
    X_scaled = scaler.transform(feat_df.values)
    y_true = (~benign_mask).astype(int).values

    # Score
    scores = model.score_samples(X_scaled)
    y_pred = (scores < threshold).astype(int)

    # Report
    print("=" * 60)
    print("VALIDATION REPORT (unseen data)")
    print("=" * 60)
    print(f"Threshold: {threshold:.6f}")
    print(f"Total:     {len(y_true)}")
    print(f"Benign:    {(y_true == 0).sum()}")
    print(f"Attack:    {(y_true == 1).sum()}\n")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  True Normal  (TN): {cm[0][0]}")
    print(f"  False Alarm  (FP): {cm[0][1]}")
    print(f"  Missed Attack(FN): {cm[1][0]}")
    print(f"  True Attack  (TP): {cm[1][1]}")
    fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    print(f"  False Positive Rate: {fpr:.4f}\n")

    # Per-attack-type
    print("=== Per Attack Type ===")
    for label in sorted(val_df["attack_label"].unique()):
        if label == BENIGN_LABEL:
            continue
        mask = (val_df["attack_label"] == label).values
        detected = int((y_pred[mask] == 1).sum())
        total = int(mask.sum())
        rate = detected / total if total > 0 else 0
        print(f"  {label:<15} {detected:>6}/{total:<6} detected ({rate*100:.1f}%)")

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc*100:.1f}%  {'✅ PASS' if acc >= 0.85 else '❌ BELOW TARGET'}")

    # Verify all model files are consistent
    print("\n=== Model File Consistency Check ===")
    device_classes = ["cctv", "router", "door_lock", "smart_light", "laptop"]
    ref_scores = scores[:10]
    for dc in device_classes:
        m = joblib.load(MODEL_DIR / f"if_{dc}.joblib")
        s = m.score_samples(X_scaled[:10])
        match = np.allclose(ref_scores, s)
        size_mb = (MODEL_DIR / f"if_{dc}.joblib").stat().st_size / 1024 / 1024
        print(f"  if_{dc}.joblib: {size_mb:.1f} MB, scores match reference: {'✅' if match else '❌'}")


if __name__ == "__main__":
    main()
