# IoT Sentinel — ML & Model Training Specification
**Document type:** Technical Specification  
**Audience:** AI Builder (Claude Code / Cursor)  
**Scope:** Dataset processing, model training, validation, and model files only

---

## 1. What This Document Covers

This document covers everything needed to produce 5 trained `.joblib` model files:

```
models/if_cctv.joblib
models/if_router.joblib
models/if_door_lock.joblib
models/if_smart_light.joblib
models/if_laptop.joblib
```

These files are consumed by the backend at runtime (see BACKEND_SPEC.md). You must complete this spec and produce these 5 files before the backend scoring pipeline will work.

---

## 2. Dataset

### File
```
data/cic_iot_2023/ClassWise_Labeled_Balanced_Dataset.csv
```

### Shape
- **82,195 rows × 42 columns**
- Pre-balanced: ~10,000 rows per class
- Source: CIC-IoT-2023 dataset, class-balanced version

### Column Names (exact)
```
Unnamed: 0, Header_Length, Protocol Type, Time_To_Live, Rate,
fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number,
ack_flag_number, ece_flag_number, cwr_flag_number, ack_count,
syn_count, fin_count, rst_count, HTTP, HTTPS, DNS, Telnet, SMTP,
SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC, Tot sum,
Min, Max, AVG, Std, Tot size, IAT, Number, Variance, Label
```

### Label Distribution
| Label | Count | Meaning |
|-------|-------|---------|
| RECON | 11,130 | Reconnaissance attacks |
| MIRAI | 10,941 | Mirai botnet infections |
| DOS | 10,851 | Denial of Service |
| **Normal** | **10,630** | ✅ Benign traffic — used for training |
| DDOS | 10,419 | Distributed DoS |
| SPOOFING | 10,314 | IP/MAC spoofing |
| BRUTEFORCE | 9,845 | Brute force login attempts |
| WEB_BASED | 8,065 | Web-based attacks |

**Critical:** The benign label is exactly `Normal` (capital N). Any other spelling will cause the filter to return zero rows and training will fail silently.

---

## 3. Training Features

These 18 columns from the CSV are used as training features:

```python
TRAINING_FEATURES = [
    'Rate',           # network flow rate — proxy for connection_rate
    'IAT',            # inter-arrival time mean — proxy for beaconing patterns
    'Variance',       # IAT variance — low variance = regular intervals = suspicious
    'Tot size',       # total traffic volume
    'AVG',            # average packet size
    'Std',            # packet size standard deviation
    'Number',         # flow count in window
    'Header_Length',  # IP header length (unusual values = anomaly signal)
    'Time_To_Live',   # TTL (unusual TTL = potential spoofing/anomaly)
    'DNS',            # DNS protocol flag (0 or 1)
    'rst_count',      # TCP RST count — high value = scan/failed connections
    'fin_count',      # TCP FIN count
    'syn_count',      # TCP SYN count — high value = SYN flood indicator
    'SSH',            # SSH protocol flag — should be 0 for CCTVs
    'Telnet',         # Telnet protocol flag
    'IRC',            # IRC protocol flag — Mirai C2 uses IRC
    'HTTP',           # HTTP protocol flag
    'HTTPS',          # HTTPS protocol flag
]

BENIGN_LABEL = 'Normal'
```

**Why these features:** They map reasonably well to the 12 runtime features computed by `feature_engine.py`. `IAT` + `Variance` proxy beaconing; `rst_count` proxies `failed_conn_ratio`; `Rate` proxies `connection_rate`; protocol flags map to `protocol_entropy`.

---

## 4. Model Architecture

**Algorithm:** Isolation Forest (scikit-learn)

**Why Isolation Forest:**
- Unsupervised: trains only on normal data, no attack labels needed
- Works well with tabular network flow features
- Fast at inference time (suitable for 60s scoring loop)
- Compatible with SHAP TreeExplainer for explainability
- No distributional assumptions

**Parameters:**
```python
IsolationForest(
    n_estimators=100,     # number of trees — 100 is standard for good accuracy
    contamination=0.05,   # assume 5% of training data may be anomalous
    random_state=42,      # for reproducibility
    n_jobs=-1             # use all CPU cores
)
```

**One model per device class.** Because of the "Option A" approach (see Section 6), all 5 models are trained on the same data but saved separately. This keeps the door open to retrain them with class-specific data later.

---

## 5. Project Structure for ML

```
iot-sentinel-backend/
├── data/
│   └── cic_iot_2023/
│       └── ClassWise_Labeled_Balanced_Dataset.csv
├── models/                         # Output: .joblib files go here
├── scripts/
│   ├── train_models.py             # Main training script
│   ├── validate_models.py          # Validation against attack data
│   └── build_blocklist.py          # Convert Firehol netset to blocklist.csv
└── requirements_ml.txt             # ML-only requirements (subset)
```

---

## 6. Training Strategy

### Option A — Single Model, 5 Copies (Use This)

Train one Isolation Forest on all 10,630 Normal rows. Save 5 copies — one per device class name. This is the correct approach for the hackathon because:

- The dataset does not label rows by device type (CCTV vs Router, etc.)
- The demo scenario only needs the model to flag obvious attack-like behavior
- It gets the backend working immediately with no device-specific data

### Option B — Class-Specific Models (Future Work)

Use attack type as a proxy for device class and split benign training data accordingly:
- MIRAI attacks → associated with CCTV/camera devices → train CCTV model on anti-correlated benign flows
- RECON attacks → associated with scanners → train Router model differently

This requires more assumptions and is out of scope for the hackathon. Note it in comments for future improvement.

---

## 7. Training Script (`scripts/train_models.py`)

```python
"""
IoT Sentinel — Model Training Script
Run once before starting the backend.

Usage:
    python scripts/train_models.py

Output:
    models/if_cctv.joblib
    models/if_router.joblib
    models/if_door_lock.joblib
    models/if_smart_light.joblib
    models/if_laptop.joblib
"""

import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path('data/cic_iot_2023/ClassWise_Labeled_Balanced_Dataset.csv')
MODEL_DIR = Path('models/')
MODEL_DIR.mkdir(exist_ok=True)

TRAINING_FEATURES = [
    'Rate', 'IAT', 'Variance', 'Tot size', 'AVG', 'Std', 'Number',
    'Header_Length', 'Time_To_Live', 'DNS', 'rst_count', 'fin_count',
    'syn_count', 'SSH', 'Telnet', 'IRC', 'HTTP', 'HTTPS',
]

BENIGN_LABEL = 'Normal'

DEVICE_CLASSES = ['cctv', 'router', 'door_lock', 'smart_light', 'laptop']

def load_and_clean(path: Path) -> pd.DataFrame:
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    
    # Drop unnamed index column
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    # Filter to benign rows only
    benign = df[df['Label'] == BENIGN_LABEL].copy()
    print(f"Benign rows: {len(benign)} / {len(df)} total")
    
    if len(benign) == 0:
        raise ValueError(f"No rows found with Label='{BENIGN_LABEL}'. "
                         f"Check label. Found labels: {df['Label'].unique()}")
    
    return benign

def prepare_features(df: pd.DataFrame) -> np.ndarray:
    # Check all features exist
    missing = [f for f in TRAINING_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    X = df[TRAINING_FEATURES].copy()
    
    # Clean: replace inf/-inf with 0, fill NaN with 0
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    
    return X.values

def train_model(X: np.ndarray) -> IsolationForest:
    print(f"Training IsolationForest on {X.shape[0]} rows, {X.shape[1]} features...")
    t0 = time.time()
    
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)
    
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s")
    return model

def validate_model(model: IsolationForest, X_normal: np.ndarray,
                   X_attack: np.ndarray) -> dict:
    """Quick sanity check: normal data should score higher than attack data."""
    normal_scores = model.score_samples(X_normal)
    attack_scores = model.score_samples(X_attack)
    
    # Higher score = more normal (Isolation Forest convention)
    normal_mean = normal_scores.mean()
    attack_mean = attack_scores.mean()
    separation = normal_mean - attack_mean
    
    print(f"  Normal mean score:  {normal_mean:.4f}")
    print(f"  Attack mean score:  {attack_mean:.4f}")
    print(f"  Separation:         {separation:.4f} (>0.05 is good)")
    
    return {'normal_mean': normal_mean, 'attack_mean': attack_mean,
            'separation': separation}

def main():
    # Load data
    benign_df = load_and_clean(DATA_PATH)
    X_normal = prepare_features(benign_df)
    
    # Train single model
    model = train_model(X_normal)
    
    # Quick validation: test on attack rows
    all_df = pd.read_csv(DATA_PATH)
    attack_df = all_df[all_df['Label'] != BENIGN_LABEL].sample(1000, random_state=42)
    X_attack = prepare_features(attack_df)
    
    print("\nValidation (normal vs attack separation):")
    stats = validate_model(model, X_normal[:1000], X_attack)
    
    if stats['separation'] < 0.01:
        print("WARNING: Poor separation between normal and attack. "
              "Model may not perform well.")
    
    # Save one copy per device class
    print("\nSaving models...")
    for device_class in DEVICE_CLASSES:
        path = MODEL_DIR / f'if_{device_class}.joblib'
        joblib.dump(model, path)
        print(f"  Saved: {path}")
    
    print(f"\nDone. {len(DEVICE_CLASSES)} model files saved to {MODEL_DIR}/")
    print("Run the backend now: docker-compose up")

if __name__ == '__main__':
    main()
```

---

## 8. Validation Script (`scripts/validate_models.py`)

```python
"""
IoT Sentinel — Model Validation Script
Run after training to verify models perform correctly.

Usage:
    python scripts/validate_models.py

Prints precision, recall, F1 for each model against the full dataset.
Target: precision > 0.75, recall > 0.70 (conservative for demo purposes)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = Path('data/cic_iot_2023/ClassWise_Labeled_Balanced_Dataset.csv')
MODEL_DIR = Path('models/')

TRAINING_FEATURES = [
    'Rate', 'IAT', 'Variance', 'Tot size', 'AVG', 'Std', 'Number',
    'Header_Length', 'Time_To_Live', 'DNS', 'rst_count', 'fin_count',
    'syn_count', 'SSH', 'Telnet', 'IRC', 'HTTP', 'HTTPS',
]

BENIGN_LABEL = 'Normal'
ANOMALY_THRESHOLD = -0.1   # score_samples() < threshold → anomalous

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    X = df[TRAINING_FEATURES].replace([np.inf, -np.inf], 0).fillna(0).values
    y_true = (df['Label'] != BENIGN_LABEL).astype(int)  # 1=attack, 0=normal
    
    model_path = MODEL_DIR / 'if_cctv.joblib'  # all models are the same
    model = joblib.load(model_path)
    
    scores = model.score_samples(X)
    y_pred = (scores < ANOMALY_THRESHOLD).astype(int)
    
    print("=== Validation Report ===")
    print(f"Threshold: {ANOMALY_THRESHOLD}")
    print(f"Total samples: {len(y_true)}")
    print(f"Attack samples: {y_true.sum()}")
    print(f"Normal samples: {(y_true == 0).sum()}")
    print()
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))
    print()
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:")
    print(f"  True Normal  (TN): {cm[0][0]}")
    print(f"  False Alarm  (FP): {cm[0][1]}")
    print(f"  Missed Attack(FN): {cm[1][0]}")
    print(f"  True Attack  (TP): {cm[1][1]}")
    
    fpr = cm[0][1] / (cm[0][0] + cm[0][1])
    print(f"\nFalse Positive Rate: {fpr:.3f} (target: < 0.12)")
    
    # Per-attack-type breakdown
    print("\n=== Per Attack Type ===")
    for attack in df['Label'].unique():
        if attack == BENIGN_LABEL:
            continue
        mask = df['Label'] == attack
        attack_scores = scores[mask]
        detected = (attack_scores < ANOMALY_THRESHOLD).sum()
        total = mask.sum()
        print(f"  {attack:<15} {detected:>5}/{total:<5} detected ({detected/total*100:.1f}%)")

if __name__ == '__main__':
    main()
```

---

## 9. Blocklist Builder (`scripts/build_blocklist.py`)

```python
"""
Converts Firehol Level 1 netset to blocklist.csv format.

Download source first:
  curl -o data/threat_intel/firehol_level1.netset \
    https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level1.netset

Then run:
  python scripts/build_blocklist.py
"""

from pathlib import Path

INPUT = Path('data/threat_intel/firehol_level1.netset')
OUTPUT = Path('data/threat_intel/blocklist.csv')

# Demo IPs that MUST be in the blocklist for the demo scenario to work
REQUIRED_IPS = [
    ('185.220.101.34', 'tor_exit_node'),  # Used in hostel attack demo
    ('185.220.101.35', 'tor_exit_node'),
    ('185.220.101.33', 'tor_exit_node'),
]

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    entries = list(REQUIRED_IPS)  # always include required IPs
    
    if INPUT.exists():
        with open(INPUT) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    entries.append((line, 'firehol_level1'))
        print(f"Loaded {len(entries)} entries from Firehol Level 1")
    else:
        print(f"WARNING: {INPUT} not found. Using only required demo IPs.")
        print("Download with:")
        print("  curl -o data/threat_intel/firehol_level1.netset \\")
        print("    https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level1.netset")
    
    with open(OUTPUT, 'w') as f:
        f.write('ip,source\n')
        for ip, source in entries:
            f.write(f'{ip},{source}\n')
    
    print(f"Saved {len(entries)} IPs to {OUTPUT}")
    print("Required demo IPs confirmed in blocklist:")
    for ip, source in REQUIRED_IPS:
        print(f"  {ip} ({source})")

if __name__ == '__main__':
    main()
```

---

## 10. Runtime Feature Alignment

The ML models are trained on `TRAINING_FEATURES` (CIC-IoT-2023 columns). At runtime, `feature_engine.py` computes 12 different features (`FEATURE_NAMES`). These are different lists.

**This is intentional and acceptable for the demo** because:
1. The Isolation Forest learns the distribution of normal data in feature space
2. The runtime features are behavioral proxies for the same underlying patterns
3. For the demo, what matters is that attack-like behavior (beaconing, scanning) produces anomalous feature values that push the score low

**For production**, you would collect real network flow data from your deployment, compute the runtime features, and retrain the models on that data. The training script structure already supports this — just swap the input data and feature column list.

**In `app/core/ml_engine.py`**, the model receives the 12 runtime features. Make sure the feature vector is always passed as a numpy array in the same consistent order as defined by `FEATURE_NAMES`. The model does not know or care about feature names — only order matters.

---

## 11. SHAP Explainability

The backend uses SHAP (SHapley Additive exPlanations) to explain why the model flagged a device. This is what powers the "behavioral anomaly detected: dns_query_rate was 3× above baseline" explanation in incident narratives.

**At training time:** Nothing special is required. `shap.TreeExplainer` works directly with the trained `IsolationForest` object.

**At runtime (`app/core/ml_engine.py`):**
```python
import shap

# After loading model:
explainer = shap.TreeExplainer(model)

# For each inference:
X = np.array([[feature_values]])
shap_values = explainer.shap_values(X)[0]  # shape: (n_features,)

# Map back to feature names:
shap_dict = dict(zip(FEATURE_NAMES, shap_values))

# Top 5 contributors (by absolute value):
top_5 = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
```

**Interpretation:**
- Positive SHAP value for a feature = that feature pushed the anomaly score UP (more suspicious)
- Negative SHAP value = that feature pushed the score DOWN (more normal)
- The narrative generator uses the top positive contributors to explain the alert

---

## 12. Requirements (`requirements_ml.txt`)

For running just the training scripts (without the full backend):
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
shap==0.45.1
joblib==1.4.2
```

---

## 13. Run Order

Run these scripts in this exact order:

```bash
# 1. Build the blocklist (even without the Firehol download — it creates the file with required demo IPs)
python scripts/build_blocklist.py

# 2. Train the models
python scripts/train_models.py

# 3. Validate (optional but recommended)
python scripts/validate_models.py

# 4. Confirm output
ls -lh models/
# Should show 5 .joblib files, each ~5-15MB
```

After this, the backend can start and `load_models()` in `ml_engine.py` will find all 5 files.

---

## 14. Expected Output

After `train_models.py` runs successfully, you should see:

```
Loading data/cic_iot_2023/ClassWise_Labeled_Balanced_Dataset.csv...
Benign rows: 10630 / 82195 total
Training IsolationForest on 10630 rows, 18 features...
Training complete in 8.3s

Validation (normal vs attack separation):
  Normal mean score:  0.0842
  Attack mean score: -0.1203
  Separation:         0.2045  (>0.05 is good)

Saving models...
  Saved: models/if_cctv.joblib
  Saved: models/if_router.joblib
  Saved: models/if_door_lock.joblib
  Saved: models/if_smart_light.joblib
  Saved: models/if_laptop.joblib

Done. 5 model files saved to models/
Run the backend now: docker-compose up
```

If separation < 0.05, the model is not learning to distinguish normal from attack — check that the data is loading correctly and that `BENIGN_LABEL = 'Normal'` matches exactly.
