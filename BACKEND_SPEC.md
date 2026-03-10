# IoT Sentinel — Backend Specification
**Document type:** Technical Specification  
**Audience:** AI Builder (Claude Code / Cursor)  
**Scope:** Everything except ML model training (see ML_SPEC.md for that)

---

## 1. What You Are Building

IoT Sentinel is a behavioral security system for IoT networks. It watches network traffic, scores each device's trustworthiness in real time, detects anomalies, and generates plain-English incident reports.

**Your job as the backend developer:** Build the Python server that does all the thinking. A React frontend and an Android app (built by other developers) will consume your API. You own the API contract — they connect to you, not the other way around.

**You are NOT building:**
- The React dashboard (another developer)
- The Android app (another developer)
- The ML models (see ML_SPEC.md — train those first, separately)

---

## 2. Tech Stack — Exact Versions

Do not change versions. Version mismatches cause subtle bugs with scikit-learn serialization, FastAPI async behavior, and SQLAlchemy compatibility.

```
Python              3.11.x
fastapi             0.111.0
uvicorn[standard]   0.30.0
websockets          12.0
sqlalchemy          2.0.30
alembic             1.13.1
psycopg2-binary     2.9.9
pandas              2.2.2
numpy               1.26.4
scikit-learn        1.5.0
shap                0.45.1
networkx            3.3
pyyaml              6.0.1
pydantic            2.7.1
python-dotenv       1.0.1
statsmodels         0.14.2
joblib              1.4.2
httpx               0.27.0
pytest              8.2.1
pytest-asyncio      0.23.7
PostgreSQL          16.x        (via Docker image postgres:16-alpine)
```

---

## 3. Project Structure

Create this exact folder/file layout:

```
iot-sentinel-backend/
├── app/
│   ├── main.py                    # FastAPI app init, CORS, router registration, lifespan
│   ├── config.py                  # Settings loaded from .env via pydantic
│   ├── database.py                # SQLAlchemy engine, session factory, get_db dependency
│   ├── models/
│   │   ├── __init__.py
│   │   ├── device.py              # Device ORM model
│   │   ├── flow.py                # FlowRecord ORM model
│   │   ├── feature.py             # FeatureVector ORM model
│   │   ├── incident.py            # Incident, Evidence, ClearanceEvent ORM models
│   │   └── policy.py              # PolicyRule ORM model
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── device.py              # Pydantic Device response schemas
│   │   ├── incident.py            # Pydantic Incident + Evidence schemas
│   │   ├── graph.py               # Pydantic GraphData schema
│   │   └── websocket.py           # Pydantic WebSocket event schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── devices.py             # /api/v1/devices routes
│   │   ├── incidents.py           # /api/v1/incidents routes
│   │   ├── graph.py               # /api/v1/graph route
│   │   ├── stats.py               # /api/v1/stats route
│   │   ├── maintenance.py         # /api/v1/maintenance route
│   │   ├── demo.py                # /api/v1/demo/replay + reset routes
│   │   └── websocket.py           # /ws/events WebSocket endpoint + ConnectionManager
│   ├── core/
│   │   ├── __init__.py
│   │   ├── feature_engine.py      # Compute 12 features from flow records
│   │   ├── ml_engine.py           # Load .joblib models + SHAP scoring
│   │   ├── cusum.py               # CUSUM drift detection
│   │   ├── policy_engine.py       # YAML rule loader + evaluator
│   │   ├── trust_score.py         # Trust score aggregator
│   │   ├── graph_engine.py        # NetworkX communication graph
│   │   ├── narrative.py           # Incident narrative text generator
│   │   └── threat_intel.py        # Blocklist IP/domain lookup
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingestion.py           # CSV flow ingestion + DB insert
│   │   ├── device_resolver.py     # IP → device_id resolution
│   │   └── scoring_loop.py        # Background task: full scoring cycle every 60s
│   └── demo/
│       ├── __init__.py
│       ├── scenarios.py           # Hostel attack scenario definition
│       └── replay.py              # Flow injection for demo
├── policies/
│   ├── cctv.yaml
│   ├── router.yaml
│   ├── door_lock.yaml
│   ├── smart_light.yaml
│   └── laptop.yaml
├── models/                        # Trained .joblib ML model files go here (from ML_SPEC.md)
│   ├── if_cctv.joblib
│   ├── if_router.joblib
│   ├── if_door_lock.joblib
│   ├── if_smart_light.joblib
│   └── if_laptop.joblib
├── data/
│   ├── threat_intel/
│   │   └── blocklist.csv          # Known malicious IPs
│   └── device_registry.json       # Seed data for demo devices
├── migrations/                    # Alembic migration files
│   └── versions/
├── tests/
│   ├── test_feature_engine.py
│   ├── test_cusum.py
│   ├── test_trust_score.py
│   ├── test_policy_engine.py
│   └── test_api.py
├── scripts/
│   └── seed_db.py                 # Insert demo devices into DB
├── alembic.ini
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 4. Database Schema

Run migrations with: `alembic upgrade head`

### 4.1 devices
```sql
CREATE TABLE devices (
    device_id    VARCHAR(64) PRIMARY KEY,
    device_class VARCHAR(20) NOT NULL,
    -- Allowed values: CCTV | ROUTER | DOOR_LOCK | SMART_LIGHT | LAPTOP
    vendor       VARCHAR(128),
    mac          VARCHAR(17) UNIQUE,
    current_ip   VARCHAR(45) NOT NULL,
    status       VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    -- Allowed values: ACTIVE | ISOLATED | MAINTENANCE | DECOMMISSIONED
    first_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 4.2 flow_records
```sql
CREATE TABLE flow_records (
    id           BIGSERIAL PRIMARY KEY,
    device_id    VARCHAR(64) REFERENCES devices(device_id),
    src_ip       VARCHAR(45) NOT NULL,
    dst_ip       VARCHAR(45) NOT NULL,
    src_port     INTEGER,
    dst_port     INTEGER,
    protocol     VARCHAR(10),    -- TCP | UDP | ICMP
    bytes_sent   BIGINT DEFAULT 0,
    bytes_recv   BIGINT DEFAULT 0,
    packets_sent INTEGER DEFAULT 0,
    packets_recv INTEGER DEFAULT 0,
    tcp_flags    INTEGER,
    is_internal  BOOLEAN DEFAULT false,
    dst_country  VARCHAR(2),
    timestamp    TIMESTAMPTZ NOT NULL,
    is_attack    BOOLEAN,        -- NULL if unknown
    attack_label VARCHAR(64)
);
CREATE INDEX idx_flows_device_ts ON flow_records(device_id, timestamp DESC);
CREATE INDEX idx_flows_ts ON flow_records(timestamp DESC);
```

### 4.3 feature_vectors
```sql
CREATE TABLE feature_vectors (
    id                  BIGSERIAL PRIMARY KEY,
    device_id           VARCHAR(64) REFERENCES devices(device_id),
    window_start        TIMESTAMPTZ NOT NULL,
    window_hours        INTEGER NOT NULL,   -- 1, 24, or 168
    connection_rate     FLOAT,
    unique_dst_ips      INTEGER,
    unique_ext_domains  INTEGER,
    country_entropy     FLOAT,
    protocol_entropy    FLOAT,
    port_entropy        FLOAT,
    failed_conn_ratio   FLOAT,
    beaconing_score     FLOAT,
    dns_query_rate      FLOAT,
    dns_nxdomain_ratio  FLOAT,
    traffic_asymmetry   FLOAT,
    off_hours_fraction  FLOAT,
    bytes_total         BIGINT,
    anomaly_score_if    FLOAT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_fv_device_window ON feature_vectors(device_id, window_start DESC, window_hours);
```

### 4.4 trust_score_history
```sql
CREATE TABLE trust_score_history (
    id               BIGSERIAL PRIMARY KEY,
    device_id        VARCHAR(64) REFERENCES devices(device_id),
    trust_score      INTEGER NOT NULL,
    risk_level       VARCHAR(20) NOT NULL,
    score_trend      VARCHAR(20) NOT NULL,
    sub_behavioral   FLOAT,
    sub_policy       FLOAT,
    sub_drift        FLOAT,
    sub_threat_intel FLOAT,
    confidence       FLOAT,
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_tsh_device_ts ON trust_score_history(device_id, computed_at DESC);
```

### 4.5 policy_rules
```sql
CREATE TABLE policy_rules (
    rule_id        VARCHAR(32) PRIMARY KEY,
    device_class   VARCHAR(20) NOT NULL,
    description    TEXT NOT NULL,
    rule_type      VARCHAR(20) NOT NULL,  -- HARD_VIOLATION | SOFT_DRIFT
    severity       VARCHAR(10) NOT NULL,  -- CRITICAL | HIGH | MEDIUM | LOW
    condition_yaml TEXT NOT NULL,
    action_yaml    TEXT NOT NULL
);
```

### 4.6 incidents
```sql
CREATE TABLE incidents (
    incident_id        VARCHAR(32) PRIMARY KEY,
    device_id          VARCHAR(64) REFERENCES devices(device_id),
    status             VARCHAR(20) NOT NULL DEFAULT 'OPEN',
    -- Allowed values: OPEN | ACKNOWLEDGED | RESOLVED
    trust_score        INTEGER NOT NULL,
    risk_level         VARCHAR(20) NOT NULL,
    confidence         FLOAT,
    score_trend        VARCHAR(20),
    score_24h_delta    INTEGER,
    recommended_action VARCHAR(32),
    recommended_vlan   INTEGER,
    narrative          TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 4.7 evidence
```sql
CREATE TABLE evidence (
    evidence_id          VARCHAR(32) PRIMARY KEY,
    incident_id          VARCHAR(32) REFERENCES incidents(incident_id),
    device_id            VARCHAR(64) REFERENCES devices(device_id),
    evidence_type        VARCHAR(30) NOT NULL,
    -- Types: POLICY_VIOLATION | ANOMALY_SCORE | DRIFT_ALERT | THREAT_INTEL_MATCH | GRAPH_ANOMALY
    rule_id              VARCHAR(32),
    description          TEXT NOT NULL,
    severity             VARCHAR(10) NOT NULL,
    confidence           FLOAT,
    timestamp            TIMESTAMPTZ NOT NULL,
    telemetry_json       JSONB,
    shap_json            JSONB,
    drift_duration_hours FLOAT,
    cusum_value          FLOAT
);
```

### 4.8 clearance_events
```sql
CREATE TABLE clearance_events (
    id          BIGSERIAL PRIMARY KEY,
    incident_id VARCHAR(32) REFERENCES incidents(incident_id),
    device_id   VARCHAR(64) REFERENCES devices(device_id),
    analyst_id  VARCHAR(64) NOT NULL,
    reason      TEXT NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 4.9 graph_edges
```sql
CREATE TABLE graph_edges (
    id             BIGSERIAL PRIMARY KEY,
    src_device_id  VARCHAR(64) NOT NULL,
    dst_identifier VARCHAR(64) NOT NULL,  -- device_id or raw IP
    is_internal    BOOLEAN NOT NULL,
    is_baseline    BOOLEAN NOT NULL DEFAULT false,
    is_anomalous   BOOLEAN NOT NULL DEFAULT false,
    anomaly_type   VARCHAR(30),
    volume_bytes   BIGINT DEFAULT 0,
    first_seen     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX idx_graph_edge ON graph_edges(src_device_id, dst_identifier);
```

---

## 5. Core Modules

### 5.1 Feature Engine (`app/core/feature_engine.py`)

Computes 12 behavioral features from raw flow records for a given device over a time window.

```python
FEATURE_NAMES = [
    'connection_rate',    # flows per hour
    'unique_dst_ips',     # count of distinct destination IPs
    'unique_ext_domains', # count of distinct external IPs (is_internal=False)
    'country_entropy',    # Shannon entropy of dst_country distribution
    'protocol_entropy',   # Shannon entropy of protocol distribution
    'port_entropy',       # Shannon entropy of dst_port distribution
    'failed_conn_ratio',  # failed connections / total connections
    'beaconing_score',    # max absolute ACF of inter-arrival times (lag 1-20, FFT)
    'dns_query_rate',     # flows with dst_port==53 per hour
    'dns_nxdomain_ratio', # 0.0 placeholder (requires Zeek)
    'traffic_asymmetry',  # bytes_sent.sum() / (bytes_recv.sum() + 1)
    'off_hours_fraction', # fraction of flows between 22:00-06:00
]
```

**Function signatures:**

```python
def shannon_entropy(counts: pd.Series) -> float:
    # Returns -sum(p * log2(p + 1e-10)) where p = proportions
    # Returns 0.0 if counts is empty

def compute_beaconing(timestamps) -> float:
    # Uses statsmodels acf(inter_arrival_times, nlags=20, fft=True)
    # Returns max(abs(autocorr[1:])) — high value means regular intervals
    # Returns 0.0 if fewer than 10 timestamps

def compute_off_hours(timestamps) -> float:
    # Returns fraction of timestamps where hour < 6 or hour >= 22

def compute_features(device_id: str, window_hours: int, db: Session) -> dict | None:
    # Returns None if fewer than 2 flow records in window
    # Queries flow_records WHERE device_id=? AND timestamp >= NOW()-window_hours
```

### 5.2 ML Engine (`app/core/ml_engine.py`)

Loads pre-trained Isolation Forest models and scores feature vectors.

**Startup:** Call `load_models()` once during FastAPI lifespan. Loads all `.joblib` files from `/models/` directory into memory. Creates a `shap.TreeExplainer` for each model.

```python
def load_models() -> None:
    # Loads if_cctv.joblib, if_router.joblib, if_door_lock.joblib,
    # if_smart_light.joblib, if_laptop.joblib
    # Creates TreeExplainer for each
    # Stores in module-level dicts: _models, _explainers

def score_features(device_class: str, feature_vector: dict) -> dict:
    # Returns:
    # {
    #   'anomaly_score': float,        # 0.0 = normal, 1.0 = fully anomalous
    #   'shap_contributions': dict     # top 5 features by absolute SHAP value
    # }
    #
    # anomaly_score formula:
    #   raw = model.score_samples(X)[0]   # typically -0.5 to +0.5
    #   anomaly_score = clip((0.5 - raw), 0, 1)
    #
    # Fallback: use CCTV model if device_class model not found
```

**Important:** The ML models were trained on CIC-IoT-2023 statistical features (see ML_SPEC.md). At runtime, the 12 features from `feature_engine.py` are passed to the same model. The feature names must match the order the model was trained on. See ML_SPEC.md for the exact `TRAINING_FEATURES` list and ensure the runtime feature vector is aligned.

### 5.3 CUSUM Engine (`app/core/cusum.py`)

Detects slow behavioral drift by accumulating deviations over time.

```python
@dataclass
class CUSUMState:
    S: float = 0.0
    onset_time: str | None = None
    drift_active: bool = False

# In-memory dict: (device_id, feature_name) -> CUSUMState
_states: dict = {}

def cusum_update(
    device_id: str,
    feature_name: str,
    value: float,
    mu0: float,       # baseline mean for this feature + device class
    sigma: float,     # baseline std dev
    delta: float = 0.5,
    threshold_h: float = 5.0,
    timestamp: str = None
) -> dict | None:
    # Normalise: x_norm = (value - mu0) / max(sigma, 1e-6)
    # Update:    S = max(0.0, S + (x_norm - delta))
    # Alert:     if S > threshold_h and not drift_active → set drift_active, return alert dict
    # Reset:     if S <= threshold_h * 0.3 → clear drift_active
    # Returns alert dict or None

def get_cusum_value(device_id: str, feature_name: str) -> float:
    # Returns current S value or 0.0

def reset_cusum(device_id: str) -> None:
    # Removes all states for this device_id
```

**Baseline values (hardcoded per device class for demo):**
```python
CUSUM_BASELINES = {
    'CCTV':        {'connection_rate': (5.0, 2.0),   'dns_query_rate': (2.0, 1.0),  'beaconing_score': (0.1, 0.05)},
    'ROUTER':      {'connection_rate': (50.0, 15.0),  'dns_query_rate': (20.0, 8.0), 'beaconing_score': (0.1, 0.05)},
    'DOOR_LOCK':   {'connection_rate': (1.0, 0.5),    'dns_query_rate': (0.5, 0.3),  'beaconing_score': (0.1, 0.05)},
    'SMART_LIGHT': {'connection_rate': (2.0, 1.0),    'dns_query_rate': (0.5, 0.3),  'beaconing_score': (0.1, 0.05)},
    'LAPTOP':      {'connection_rate': (30.0, 10.0),  'dns_query_rate': (10.0, 5.0), 'beaconing_score': (0.2, 0.1)},
}
# Format: (mu0, sigma)
```

### 5.4 Policy Engine (`app/core/policy_engine.py`)

Loads YAML rule files and evaluates them against flows and feature vectors.

```python
def load_policies(policies_dir: str = 'policies/') -> dict:
    # Returns dict: {device_class: [rule_dicts]}
    # Reads all .yaml files in the directory

def evaluate_flow(device_class: str, flow: dict, blocklist_ips: set) -> list[dict]:
    # Evaluates HARD_VIOLATION rules against a single flow event
    # Returns list of triggered rules (may be empty)
    # Supported condition keys:
    #   protocol: "TCP" | "UDP" | "ICMP"
    #   dst_port: int
    #   dst_type: "external" | "internal"
    #   dst_ip_in: "threat_intel_blocklist"

def evaluate_features(device_class: str, features: dict) -> list[dict]:
    # Evaluates SOFT_DRIFT rules against a feature vector
    # Supported condition keys:
    #   feature_gt: {feature_name: threshold}
    #   feature_lt: {feature_name: threshold}
```

### 5.5 Trust Score Calculator (`app/core/trust_score.py`)

Combines all signals into a single 0–100 trust score using geometric mean.

```python
RISK_THRESHOLDS = {
    'TRUSTED':     (80, 100),
    'LOW_RISK':    (60, 79),
    'MEDIUM_RISK': (40, 59),
    'HIGH_RISK':   (20, 39),
    'CRITICAL':    (0,  19),
}

VIOLATION_PENALTIES = {
    'CRITICAL': 0.6,
    'HIGH':     0.4,
    'MEDIUM':   0.2,
    'LOW':      0.05,
}

def compute_trust_score(
    anomaly_score: float,
    violations: list[dict],
    cusum_states: dict,        # {feature_name: CUSUMState}
    is_blacklisted: bool,
    threshold_h: float = 5.0
) -> dict:
    # Sub-scores (all 0.0–1.0):
    #   behavioral_sub  = 1.0 - anomaly_score
    #   policy_sub      = max(0.0, 1.0 - sum(VIOLATION_PENALTIES[v['severity']] for v in violations))
    #   drift_sub       = max(0.0, 1.0 - max(s.S for s in cusum_states.values()) / threshold_h)
    #   threat_intel_sub = 0.0 if is_blacklisted else 1.0
    #
    # trust_score = round(100 * (behavioral * policy * drift * threat_intel) ** 0.25)
    #
    # Returns: {'trust_score': int, 'risk_level': str, 'sub_scores': dict, 'confidence': float}

def get_risk_level(score: int) -> str:
    # Maps score to risk level string

def get_score_trend(device_id: str, current_score: int, db: Session) -> tuple[str, int]:
    # Compares to score 1 hour ago from trust_score_history
    # Returns (trend_string, delta_int)
    # trend: 'IMPROVING' if delta > +5, 'DECLINING' if delta < -5, else 'STABLE'
```

### 5.6 Graph Engine (`app/core/graph_engine.py`)

Tracks device communication patterns using NetworkX DiGraph.

```python
class GraphEngine:
    def __init__(self):
        self.graph = nx.DiGraph()

    def load_from_db(self, db: Session) -> None:
        # Restore edges from graph_edges table on startup

    def update_edge(self, src_device_id: str, dst_identifier: str,
                    is_internal: bool, bytes: int, db: Session) -> dict | None:
        # Add or update edge weight
        # If edge is NEW (never seen before) → mark is_anomalous=True, return anomaly dict
        # Persist to graph_edges table
        # Anomaly types: NEW_COMM_EDGE, CROSS_DEVICE_COMM, HIGH_VOLUME

    def mark_baseline(self, src: str, dst: str) -> None:
        # Mark edge as baseline (seen during normal operation)

    def export_graph(self) -> dict:
        # Returns {'nodes': [...], 'edges': [...]}
        # Nodes include: id, device_class, trust_score, risk_level
        # Edges include: source, target, is_internal, is_baseline, is_anomalous,
        #                volume_bytes, last_seen
```

### 5.7 Narrative Generator (`app/core/narrative.py`)

Generates plain-English incident reports from evidence lists.

```python
def generate_narrative(
    device_id: str,
    device_class: str,
    evidence_list: list[dict],
    trust_score: int,
    risk_level: str,
    recommended_action: str
) -> str:
    # Builds a 3–6 sentence narrative
    # Sentence templates by evidence_type:
    #   POLICY_VIOLATION    → "At {time}, {device_id} violated policy {rule_id}: {description}."
    #   ANOMALY_SCORE       → "Behavioral anomaly detected: {top_feature} was elevated above class baseline."
    #   DRIFT_ALERT         → "{feature} has been drifting above normal for {duration} hours."
    #   THREAT_INTEL_MATCH  → "Outbound connection to known malicious IP {dst_ip}."
    #   GRAPH_ANOMALY       → "Unexpected communication detected from {src} to {dst}."
    # Always ends with: "Trust score dropped to {score}/100 ({risk_level}). Recommended action: {action}."
```

### 5.8 Threat Intel (`app/core/threat_intel.py`)

```python
def load_blocklist(path: str = 'data/threat_intel/blocklist.csv') -> set:
    # Load CSV, return set of IP strings
    # File format: one IP per line, with optional header

def is_malicious(ip: str, blocklist: set) -> bool:
    # Exact match lookup. Returns True if IP in blocklist.

def get_intel_label(ip: str) -> str | None:
    # Return source label string if available, else None
```

**Minimum blocklist requirement for demo:** The IP `185.220.101.34` (Tor exit node used in the demo scenario) **must** be in the blocklist. Add it manually if not present.

---

## 6. Pipeline

### 6.1 Device Resolver (`app/pipeline/device_resolver.py`)

```python
class DeviceResolver:
    def __init__(self):
        self._cache: dict = {}  # ip -> device_id

    def load_from_db(self, db: Session) -> None:
        # Populate cache from devices table: {current_ip: device_id}

    def resolve(self, src_ip: str) -> str | None:
        # Return device_id for this IP, or None if not found
```

### 6.2 Ingestion (`app/pipeline/ingestion.py`)

```python
def ingest_csv(filepath: str, db: Session, resolver: DeviceResolver) -> dict:
    # Reads CSV, maps columns to FlowRecord schema, batch inserts (size 1000)
    # Skips rows where src_ip not in resolver cache
    # Returns {'ingested': int, 'skipped': int}
```

### 6.3 Scoring Loop (`app/pipeline/scoring_loop.py`)

This is the heart of the system. Runs as an asyncio background task every 60 seconds.

```python
async def scoring_loop(db_factory, ws_manager: ConnectionManager) -> None:
    while True:
        await asyncio.sleep(60)
        db = db_factory()
        try:
            devices = get_active_devices(db)
            for device in devices:
                await score_device(device, db, ws_manager)
        finally:
            db.close()

async def score_device(device, db, ws_manager) -> None:
    # Step 1: Compute features
    features = compute_features(device.device_id, window_hours=1, db=db)
    if features is None:
        return

    # Step 2: ML scoring
    ml_result = score_features(device.device_class, features)

    # Step 3: Store feature vector to DB
    save_feature_vector(device.device_id, features, ml_result['anomaly_score'], db)

    # Step 4: Policy evaluation (against last 60s of flows)
    recent_flows = get_recent_flows(device.device_id, seconds=60, db=db)
    violations = []
    for flow in recent_flows:
        violations += evaluate_flow(device.device_class, flow, blocklist)

    # Step 5: CUSUM update
    baselines = CUSUM_BASELINES.get(device.device_class, {})
    for feature_name, (mu0, sigma) in baselines.items():
        alert = cusum_update(device.device_id, feature_name,
                             features.get(feature_name, 0), mu0, sigma)
        if alert:
            # Create DRIFT_ALERT evidence

    # Step 6: Threat intel check
    external_ips = get_recent_external_ips(device.device_id, seconds=60, db=db)
    is_blacklisted = any(is_malicious(ip, blocklist) for ip in external_ips)

    # Step 7: Trust score
    cusum_states = get_all_cusum_states(device.device_id)
    result = compute_trust_score(ml_result['anomaly_score'], violations,
                                 cusum_states, is_blacklisted)

    # Step 8: Persist trust score
    save_trust_score(device.device_id, result, db)

    # Step 9: Create incident if score crossed risk boundary downward
    prev_score = get_previous_trust_score(device.device_id, db)
    if crossed_boundary_downward(prev_score, result['trust_score']):
        evidence = build_evidence_list(violations, ml_result, cusum_states, is_blacklisted)
        narrative = generate_narrative(device.device_id, device.device_class,
                                       evidence, result['trust_score'],
                                       result['risk_level'], get_recommended_action(result))
        incident = create_incident(device, result, evidence, narrative, db)
        await ws_manager.broadcast({'event': 'new_incident', ...})

    # Step 10: WebSocket push
    await ws_manager.broadcast({'event': 'trust_score_update',
                                 'device_id': device.device_id,
                                 'trust_score': result['trust_score'], ...})
```

**Risk boundary crossing logic:**
A boundary is crossed when score moves from one risk band to a worse one:
- TRUSTED (80–100) → LOW_RISK (60–79) → MEDIUM_RISK (40–59) → HIGH_RISK (20–39) → CRITICAL (0–19)
- Only downward crossings create incidents (score getting worse)
- Don't create duplicate incidents if device already has an OPEN incident

---

## 7. Policy YAML Files

### `policies/cctv.yaml`
```yaml
policies:
  - rule_id: CAM_001
    description: CCTV must not initiate external SSH connections
    type: HARD_VIOLATION
    severity: CRITICAL
    condition:
      protocol: TCP
      dst_port: 22
      dst_type: external
    action: {alert: true, recommended: VLAN_ISOLATION}

  - rule_id: CAM_002
    description: CCTV must not contact blacklisted IPs
    type: HARD_VIOLATION
    severity: CRITICAL
    condition:
      dst_ip_in: threat_intel_blocklist
    action: {alert: true, recommended: VLAN_ISOLATION}

  - rule_id: CAM_003
    description: CCTV must not contact Tor exit nodes
    type: HARD_VIOLATION
    severity: CRITICAL
    condition:
      dst_ip_in: threat_intel_blocklist
    action: {alert: true, recommended: VLAN_ISOLATION}

  - rule_id: CAM_004
    description: CCTV DNS query rate exceeds 3x class baseline
    type: SOFT_DRIFT
    severity: HIGH
    condition:
      feature_gt: {dns_query_rate: 6.0}
    action: {alert: true, recommended: MONITOR_ELEVATED}

  - rule_id: CAM_005
    description: CCTV initiating connections to new destination IPs (scanning behaviour)
    type: HARD_VIOLATION
    severity: HIGH
    condition:
      dst_type: internal
      new_destination: true
    action: {alert: true, recommended: MONITOR_ELEVATED}
```

### `policies/router.yaml`
```yaml
policies:
  - rule_id: RTR_001
    description: Router must not contact blacklisted IPs
    type: HARD_VIOLATION
    severity: CRITICAL
    condition:
      dst_ip_in: threat_intel_blocklist
    action: {alert: true, recommended: VLAN_ISOLATION}

  - rule_id: RTR_002
    description: Unusual spike in unique destination IPs
    type: SOFT_DRIFT
    severity: HIGH
    condition:
      feature_gt: {unique_dst_ips: 200}
    action: {alert: true, recommended: MONITOR_ELEVATED}

  - rule_id: RTR_003
    description: Router SSH to external IP
    type: HARD_VIOLATION
    severity: HIGH
    condition:
      protocol: TCP
      dst_port: 22
      dst_type: external
    action: {alert: true, recommended: INVESTIGATE}
```

### `policies/door_lock.yaml`
```yaml
policies:
  - rule_id: DLK_001
    description: Door lock must not initiate any external connections
    type: HARD_VIOLATION
    severity: CRITICAL
    condition:
      dst_type: external
    action: {alert: true, recommended: VLAN_ISOLATION}

  - rule_id: DLK_002
    description: Door lock must not contact blacklisted IPs
    type: HARD_VIOLATION
    severity: CRITICAL
    condition:
      dst_ip_in: threat_intel_blocklist
    action: {alert: true, recommended: VLAN_ISOLATION}
```

### `policies/smart_light.yaml`
```yaml
policies:
  - rule_id: SLT_001
    description: Smart light must not contact blacklisted IPs
    type: HARD_VIOLATION
    severity: CRITICAL
    condition:
      dst_ip_in: threat_intel_blocklist
    action: {alert: true, recommended: VLAN_ISOLATION}

  - rule_id: SLT_002
    description: Smart light making excessive external connections
    type: SOFT_DRIFT
    severity: MEDIUM
    condition:
      feature_gt: {unique_ext_domains: 5}
    action: {alert: true, recommended: MONITOR_ELEVATED}
```

### `policies/laptop.yaml`
```yaml
policies:
  - rule_id: LPT_001
    description: Laptop contacted blacklisted IP
    type: HARD_VIOLATION
    severity: HIGH
    condition:
      dst_ip_in: threat_intel_blocklist
    action: {alert: true, recommended: INVESTIGATE}

  - rule_id: LPT_002
    description: Unusual port entropy — may indicate port scanning
    type: SOFT_DRIFT
    severity: MEDIUM
    condition:
      feature_gt: {port_entropy: 4.5}
    action: {alert: true, recommended: MONITOR_ELEVATED}
```

---

## 8. Seed Data (`data/device_registry.json`)

```json
[
  {
    "device_id": "CCTV_01",
    "device_class": "CCTV",
    "vendor": "Hikvision",
    "mac": "AA:BB:CC:DD:EE:01",
    "current_ip": "192.168.1.45",
    "status": "ACTIVE"
  },
  {
    "device_id": "CCTV_02",
    "device_class": "CCTV",
    "vendor": "Dahua",
    "mac": "AA:BB:CC:DD:EE:02",
    "current_ip": "192.168.1.46",
    "status": "ACTIVE"
  },
  {
    "device_id": "ROUTER_01",
    "device_class": "ROUTER",
    "vendor": "Cisco",
    "mac": "AA:BB:CC:DD:EE:03",
    "current_ip": "192.168.1.1",
    "status": "ACTIVE"
  },
  {
    "device_id": "DOOR_01",
    "device_class": "DOOR_LOCK",
    "vendor": "HID Global",
    "mac": "AA:BB:CC:DD:EE:04",
    "current_ip": "192.168.1.20",
    "status": "ACTIVE"
  },
  {
    "device_id": "LAPTOP_01",
    "device_class": "LAPTOP",
    "vendor": "Dell",
    "mac": "AA:BB:CC:DD:EE:05",
    "current_ip": "192.168.1.100",
    "status": "ACTIVE"
  }
]
```

---

## 9. REST API Contract

**Base URL:** `http://localhost:8000/api/v1`  
**All responses:** JSON  
**All timestamps:** ISO 8601 with timezone (e.g. `2024-03-15T02:47:00Z`)

### 9.1 Devices

| Method | Path | Description |
|--------|------|-------------|
| GET | `/devices` | List all devices sorted by trust_score ASC |
| GET | `/devices/{device_id}` | Single device with current trust score |
| GET | `/devices/{device_id}/history?hours=24` | Trust score history as time series |
| PATCH | `/devices/{device_id}/status` | Change device status (ISOLATED, ACTIVE, etc.) |

**PATCH `/devices/{device_id}/status` body:**
```json
{"status": "ISOLATED", "reason": "SOC analyst isolated", "analyst_id": "soc_01"}
```

### 9.2 Incidents

| Method | Path | Description |
|--------|------|-------------|
| GET | `/incidents?status=OPEN&limit=50&offset=0` | Paginated incident list |
| GET | `/incidents/{incident_id}` | Single incident with evidence + clearance_events |
| PATCH | `/incidents/{incident_id}/acknowledge` | Set status = ACKNOWLEDGED |
| POST | `/incidents/{incident_id}/clear` | Set status = RESOLVED, add clearance event |

**POST `/incidents/{incident_id}/clear` body:**
```json
{"reason": "False positive — firmware update", "analyst_id": "soc_01"}
```

### 9.3 Other Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/graph?hours=1` | Communication graph (nodes + edges) |
| GET | `/stats` | Dashboard header stats |
| POST | `/maintenance/toggle` | Pause/resume scoring loop |
| GET | `/health` | Health check |
| POST | `/demo/replay` | Start hostel attack demo scenario |
| POST | `/demo/reset` | Reset all demo data to clean state |

**POST `/demo/replay` body:**
```json
{"scenario": "hostel_attack", "speed_multiplier": 1.0}
```

### 9.4 Response Shapes

**Device object:**
```json
{
  "device_id": "CCTV_01",
  "device_class": "CCTV",
  "vendor": "Hikvision",
  "mac": "AA:BB:CC:DD:EE:01",
  "current_ip": "192.168.1.45",
  "status": "ACTIVE",
  "first_seen": "2024-03-01T00:00:00Z",
  "last_seen": "2024-03-15T02:47:00Z",
  "trust_score": 23,
  "risk_level": "HIGH_RISK",
  "score_trend": "DECLINING",
  "score_24h_delta": -38,
  "confidence": 0.91,
  "sub_scores": {
    "behavioral": 0.21,
    "policy": 0.00,
    "drift": 0.85,
    "threat_intel": 0.00
  },
  "active_incident_id": "INC-2024-0315-001"
}
```

**Incident object:**
```json
{
  "incident_id": "INC-2024-0315-001",
  "device_id": "CCTV_01",
  "device_class": "CCTV",
  "vendor": "Hikvision",
  "current_ip": "192.168.1.45",
  "created_at": "2024-03-15T02:47:00Z",
  "updated_at": "2024-03-15T02:47:00Z",
  "status": "OPEN",
  "trust_score": 23,
  "risk_level": "HIGH_RISK",
  "confidence": 0.91,
  "score_trend": "DECLINING",
  "score_24h_delta": -38,
  "evidence": [],
  "recommended_action": "VLAN_ISOLATION",
  "recommended_vlan": 99,
  "adjacent_devices_at_risk": ["CCTV_02"],
  "narrative": "At 01:15 AM, CCTV_01 initiated an unusual connection...",
  "clearance_events": []
}
```

**Stats object:**
```json
{
  "total_devices": 5,
  "trusted_count": 2,
  "low_risk_count": 1,
  "medium_risk_count": 0,
  "high_risk_count": 1,
  "critical_count": 0,
  "active_incidents": 2,
  "total_incidents_today": 3,
  "maintenance_mode": false,
  "pipeline_status": "RUNNING",
  "last_updated": "2024-03-15T02:47:00Z"
}
```

**GraphData object:**
```json
{
  "nodes": [
    {"id": "CCTV_01", "device_class": "CCTV", "trust_score": 23, "risk_level": "HIGH_RISK"},
    {"id": "CCTV_02", "device_class": "CCTV", "trust_score": 78, "risk_level": "LOW_RISK"}
  ],
  "edges": [
    {
      "source": "CCTV_01", "target": "192.168.1.200",
      "is_internal": true, "is_baseline": true, "is_anomalous": false,
      "volume_bytes": 1048576, "last_seen": "2024-03-15T02:47:00Z"
    }
  ]
}
```

**TrustScoreHistory object:**
```json
{
  "device_id": "CCTV_01",
  "hours": 24,
  "data_points": [
    {"timestamp": "2024-03-14T02:47:00Z", "trust_score": 94, "risk_level": "TRUSTED"},
    {"timestamp": "2024-03-14T03:47:00Z", "trust_score": 61, "risk_level": "LOW_RISK"}
  ]
}
```

---

## 10. WebSocket

**Endpoint:** `ws://localhost:8000/ws/events`  
**Auth:** None required  
**On connect:** Server immediately sends one `system_stats` event

### Connection Manager
```python
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None
    async def disconnect(self, websocket: WebSocket) -> None
    async def broadcast(self, message: dict) -> None
```

### Event Payloads

```json
// trust_score_update — every 5s per device, or immediately if delta > 5 pts
{"event": "trust_score_update", "device_id": "CCTV_01", "trust_score": 23,
 "risk_level": "HIGH_RISK", "score_trend": "DECLINING", "score_24h_delta": -38,
 "timestamp": "2024-03-15T02:47:00Z"}

// new_incident
{"event": "new_incident", "incident_id": "INC-2024-0315-001",
 "device_id": "CCTV_01", "device_class": "CCTV", "risk_level": "HIGH_RISK",
 "trust_score": 23, "summary": "Device showing C2 beaconing + Tor contact",
 "recommended_action": "VLAN_ISOLATION", "timestamp": "..."}

// policy_violation — fired immediately when a hard rule triggers
{"event": "policy_violation", "device_id": "CCTV_01", "rule_id": "CAM_002",
 "severity": "CRITICAL", "description": "Outbound connection to blacklisted IP 185.220.101.34",
 "timestamp": "..."}

// graph_anomaly
{"event": "graph_anomaly", "src_device": "CCTV_01", "dst_device": "CCTV_02",
 "anomaly_type": "NEW_COMM_EDGE", "severity": "HIGH", "timestamp": "..."}

// device_status_change
{"event": "device_status_change", "device_id": "CCTV_01",
 "old_status": "ACTIVE", "new_status": "ISOLATED",
 "reason": "SOC analyst initiated isolation", "timestamp": "..."}

// system_stats — every 10s
{"event": "system_stats", "total_devices": 5, "trusted_count": 2,
 "low_risk_count": 1, "medium_risk_count": 0, "high_risk_count": 1,
 "critical_count": 0, "active_incidents": 2, "maintenance_mode": false,
 "timestamp": "..."}

// demo_progress — during demo replay
{"event": "demo_progress", "scenario": "hostel_attack", "step": 3,
 "total_steps": 6, "message": "CCTV_01 contacting Tor exit node — trust score dropping",
 "timestamp": "..."}
```

---

## 11. Demo Replay System

### Scenario: `hostel_attack`

6 steps. Each step injects pre-defined flows into `flow_records`, then triggers a scoring cycle.

| Step | Delay | Device | Injected Flows | Expected Trust Score |
|------|-------|--------|----------------|---------------------|
| 1 | 0s | CCTV_01 | 20 normal flows to 192.168.1.200:554 (RTSP), TCP, ~50KB each | ~94 |
| 2 | 15s | CCTV_01 | 10 flows to 192.168.1.10, .11, .12 — new IPs, TCP SYN pattern | ~61 |
| 3 | 90s | CCTV_01 | 1 flow to 185.220.101.34:443 (blacklisted Tor exit node) | ~29 |
| 4 | 120s | CCTV_01 | 15 flows to 185.220.101.34 at exactly 30s intervals (beaconing) | ~17 |
| 5 | 125s | CCTV_02 | 8 DNS flows (dst_port=53) in 1 minute — elevated rate | ~78 |
| 6 | auto | — | SOC calls clear endpoint | — |

**Demo replay API:**
- `POST /demo/replay` — starts as background asyncio task, returns immediately
- `POST /demo/reset` — deletes all incidents, evidence, resets trust scores to 90+, clears CUSUM, clears graph anomalies, returns `{"status": "reset_complete"}`

---

## 12. Docker Setup

### `docker-compose.yml`
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: iot_sentinel
      POSTGRES_USER: sentinel
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sentinel"]
      interval: 5s
      timeout: 5s
      retries: 5

  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://sentinel:${DB_PASSWORD}@postgres:5432/iot_sentinel
      DB_PASSWORD: ${DB_PASSWORD}
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./policies:/app/policies

volumes:
  pgdata:
```

### `.env.example`
```
DATABASE_URL=postgresql://sentinel:password@localhost:5432/iot_sentinel
DB_PASSWORD=changeme
SECRET_KEY=change-this-in-production
MAINTENANCE_MODE=false
```

### `Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 13. app/main.py Structure

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup:
    load_models()               # ML engine
    load_policies()             # Policy engine
    load_blocklist()            # Threat intel
    graph_engine.load_from_db() # Graph engine
    device_resolver.load_from_db()
    asyncio.create_task(scoring_loop(db_factory, ws_manager))
    asyncio.create_task(ws_heartbeat(ws_manager))  # system_stats every 10s
    yield
    # Shutdown: nothing required

app = FastAPI(title="IoT Sentinel API", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Register routers
app.include_router(devices_router, prefix="/api/v1")
app.include_router(incidents_router, prefix="/api/v1")
app.include_router(graph_router, prefix="/api/v1")
app.include_router(stats_router, prefix="/api/v1")
app.include_router(maintenance_router, prefix="/api/v1")
app.include_router(demo_router, prefix="/api/v1")
app.include_router(ws_router)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## 14. Tests

| File | What to test |
|------|-------------|
| `test_feature_engine.py` | Features computed correctly from synthetic flow data; returns None for <2 flows; beaconing score >0.8 for perfectly periodic data |
| `test_cusum.py` | Alert fires after sustained deviation; no alert on stable data; reset works |
| `test_trust_score.py` | Score=100 for all-clean inputs; score=0 when blacklisted; geometric mean behaves correctly |
| `test_policy_engine.py` | YAML loads all 5 device classes; CAM_001 fires on SSH external; normal flow returns [] |
| `test_api.py` | All GET endpoints return 200; POST /demo/reset returns reset_complete; WebSocket receives system_stats on connect |

---

## 15. CORS & Logging

**CORS:** Allow all origins (demo/hackathon environment).

**Logging:** Use Python standard `logging`. 
- INFO: scoring loop start/end, incidents created, model load
- WARNING: policy violations, blacklist hits
- DEBUG: per-device trust scores, WebSocket messages
