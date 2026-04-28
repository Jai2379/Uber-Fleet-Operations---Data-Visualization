<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML_Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black" />
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Production-brightgreen?style=for-the-badge" />
</p>

<h1 align="center">🚗 Fleet Operations & Cancellation Prediction Engine</h1>

<p align="center">
  <b>An end-to-end Machine Learning pipeline and Business Intelligence suite designed to predict Uber ride cancellations before they happen, and visualize the financial impact of lost revenue for operations teams.</b>
</p>

<br/>

<p align="center">
  <img src="data/Dasboard.gif" alt="Power BI Dashboard Demo" width="90%" />
</p>

---

## 📋 Table of Contents

- [The Business Problem](#-the-business-problem)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Data Engineering Achievements](#-data-engineering-achievements)
- [Module Breakdown](#-module-breakdown)
- [Power BI Dashboard](#-power-bi-dashboard)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🔍 The Business Problem

Ride-sharing companies lose **massive amounts of potential revenue** and operational time when dispatched rides are cancelled — whether by the driver or the customer. Every cancelled ride is a vehicle sitting idle, a customer left waiting, and revenue that evaporates from the books.

**This project solves that problem at two levels:**

| Layer | What It Does |
|:------|:-------------|
| **Predictive (ML)** | Identifies high-risk rides *before* dispatch using a trained Random Forest model, enabling preemptive re-routing or alternative assignments. |
| **Diagnostic (BI)** | Visualizes cancellation patterns by vehicle type, time of day, and root cause — giving ops managers the *why* behind the numbers. |

> The core insight: cancellation is **not random**. It's a predictable function of vehicle type, customer history, time-of-day friction, and driver availability metrics.

---

## 📊 Key Results

| Metric | Value |
|:-------|:------|
| **Revenue Leakage Identified** | ~₹5M in incomplete/cancelled bookings (~8.84% of total GMV) |
| **Highest-Friction Vehicle Class** | Auto — highest total volume *and* highest raw cancellation count |
| **Features Selected by Model** | 8 statistically significant predictors via SelectKBest (ANOVA F-test) |
| **Class Balancing** | SMOTE oversampling applied to correct severe class imbalance in cancellation labels |
| **Dataset Scale** | 100,000+ historical ride records from NCR region |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA LAYER                               │
│   data/ncr_ride_bookings.csv  ──►  100K+ ride records (Kaggle)     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     INGESTION LAYER                                 │
│   importer.py  ──►  Reads CSV → Loads into SQLite (uber_operations)│
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ML PIPELINE LAYER                                 │
│   pipeline.py                                                       │
│   ├── Data Cleaning (null scrubbing, type coercion)                │
│   ├── Feature Engineering (one-hot encoding, datetime → hour)      │
│   ├── Feature Selection (SelectKBest, k=8)                         │
│   ├── Class Balancing (SMOTE)                                      │
│   └── Model Training (RandomForestClassifier)                      │
│        └── Exports ──► models/active_model.pkl                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌──────────────────────┐ ┌──────────────────────────────────────────┐
│   INFERENCE LAYER    │ │         VISUALIZATION LAYER              │
│ predict_cancellation │ │   Power BI Dashboard (Uber.pbix)        │
│   .py                │ │   ├── Revenue Leakage Tracker (Pie)     │
│ ┌──────────────────┐ │ │   ├── Vehicle Friction Chart (Bar)      │
│ │ check_ride_risk()│ │ │   └── Root Cause Matrix (Table)         │
│ │ → Loads .pkl     │ │ │                                          │
│ │ → Pads features  │ │ └──────────────────────────────────────────┘
│ │ → Returns prob.  │ │
│ └──────────────────┘ │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  VERIFICATION LAYER  │
│  targeted_check.py   │
│  → Queries SQLite    │
│  → Runs inference    │
│  → Validates live    │
│    connection        │
└──────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **Language** | Python 3 |
| **Data Engineering** | Pandas · NumPy |
| **Machine Learning** | Scikit-Learn (`RandomForestClassifier`, `SelectKBest`, `f_classif`) |
| **Class Balancing** | Imbalanced-Learn (`SMOTE`) |
| **Database** | SQLite3 |
| **Model Serialization** | Joblib |
| **Business Intelligence** | Power BI |

---

## ⚡ Data Engineering Achievements

The raw dataset was **heavily polluted** — a realistic scenario that mirrors production-grade data quality challenges. The pipeline includes an automated cleaning layer that handles the following:

### 🧹 Null Corruption Handling
The dataset contained **string-literal `"null"` values** instead of proper `NaN` entries. The pipeline detects and replaces these phantom nulls before any computation occurs:
```python
df = df.replace('null', np.nan)
df = df.fillna(0)
```

### 🔢 Categorical-to-Numeric Translation
Vehicle types (`Auto`, `Go Sedan`, `eBike`, etc.) are transformed via **one-hot encoding** using `pd.get_dummies()`, converting qualitative categories into binary feature columns the Random Forest can consume:
```python
df = pd.get_dummies(df, columns=['Vehicle Type'], drop_first=True)
```

### 🕐 Temporal Feature Extraction
Raw timestamp strings (`HH:MM:SS`) are parsed into continuous integer `Hour` values (0–23), enabling the model to detect **time-of-day cancellation patterns**:
```python
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
```

### 🎯 Target Binarization
Natural-language booking statuses (`"Completed"`, `"Cancelled by Customer"`, etc.) are mapped into a strict binary classification target — `0` (completed) vs. `1` (cancelled/incomplete):
```python
df[target_col] = df[target_col].apply(
    lambda x: 0 if str(x).strip().lower() in ['completed', 'success'] else 1
)
```

### ⚖️ Class Imbalance Correction
Completed rides vastly outnumber cancellations (~91% vs ~9%). **SMOTE** (Synthetic Minority Oversampling) generates synthetic cancellation examples so the model doesn't learn to blindly predict "completed" for everything:
```python
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_new, y)
```

---

## 📦 Module Breakdown

### `importer.py` — Database Initialization
Reads the raw CSV and securely loads it into a local SQLite database (`uber_operations.db`). Designed for idempotent re-runs — each execution fully replaces the previous dataset.

```
📂 loading new dataset...
🔄 swapping old data for new data in the database...
✅ database updated!
```

---

### `pipeline.py` — Automated ML Engine
The core training pipeline wrapped in an `OperationsPipeline` class. Executes the full ML workflow in a single call:

1. **Load** raw CSV data
2. **Clean** corrupted null strings and coerce numeric types
3. **Engineer** temporal and categorical features
4. **Select** top-k features via ANOVA F-test (`SelectKBest`)
5. **Balance** classes with SMOTE oversampling
6. **Train** a `RandomForestClassifier` (100 estimators, max depth 10)
7. **Serialize** the model + feature manifest to `models/active_model.pkl`

---

### `predict_cancellation.py` — Real-Time Inference
Exposes a `check_ride_risk()` function that accepts a dictionary of ride attributes, loads the trained `.pkl` model, **dynamically pads missing variables** with neutral defaults, and outputs:

- A **cancellation probability percentage**
- A binary **risk flag** (`HIGH RISK` or `CLEAR`)

```python
>>> check_ride_risk({"Hour": 18, "Avg VTAT": 12.5, "Vehicle Type_Auto": 1})
# calculated cancellation probability: 73.40%
# "flag: high risk of cancellation. dispatch alternative."
```

---

### `targeted_check.py` — Live Verification
Queries the SQLite database for a specific ride record by row ID, extracts its features, and runs it through the inference engine. Used to validate that the full data → model → prediction pipeline is connected and functional.

---

## 📈 Power BI Dashboard

<p align="center">
  <img src="dashboards/Uber Statistics.png" alt="Power BI Dashboard" width="90%" />
</p>

The BI layer connects directly to `uber_operations.db` and presents three core operational visuals:

| Visual | Type | Insight |
|:-------|:-----|:--------|
| **Revenue Leakage Tracker** | Pie Chart | Compares ₹47M in completed ride value against ₹5M in leaked/cancelled value — a clear 8.84% revenue gap |
| **Vehicle Friction Chart** | Stacked Bar | Counts rides by vehicle type, color-coded by booking status. Reveals that **Auto** and **Go Mini** carry the highest cancellation friction |
| **Root Cause Matrix** | Table | Maps vehicle types against specific driver-side cancellation reasons, enabling targeted driver retention strategies |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Power BI Desktop *(for dashboard viewing)*

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/fleet-cancellation-engine.git
cd fleet-cancellation-engine

# 2. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn joblib

# 3. Initialize the database
python importer.py

# 4. Train the model
python pipeline.py

# 5. Verify the pipeline
python targeted_check.py
```

---

## 💡 Usage

### Predict cancellation risk for a new ride

```python
from predict_cancellation import check_ride_risk

ride = {
    "Hour": 21,
    "Avg VTAT": 15.0,
    "Avg CTAT": 8.3,
    "Cancelled Rides by Customer": 3,
    "Vehicle Type_Auto": 1
}

result = check_ride_risk(ride)
print(result)
# calculated cancellation probability: 68.20%
# "flag: high risk of cancellation. dispatch alternative."
```

### Query a specific historical ride

```python
from targeted_check import analyze_ride

analyze_ride(42)
# --- analyzing operations record id: 42 ---
# calculated cancellation probability: 12.50%
# "clear: ride likely to complete."
```

---

## 🗂 Project Structure

```
fleet-cancellation-engine/
│
├── data/
│   ├── ncr_ride_bookings.csv        # Raw dataset (100K+ ride records)
│   └── Dasboard.gif                 # Dashboard demo animation
│
├── dashboards/
│   └── Uber Statistics.png          # Dashboard static screenshot
│
├── models/
│   └── active_model.pkl             # Serialized trained model + feature manifest
│
├── importer.py                      # CSV → SQLite database loader
├── pipeline.py                      # Full ML training pipeline
├── predict_cancellation.py          # Real-time inference engine
├── targeted_check.py                # Live pipeline verification script
├── uber_operations.db               # SQLite operational database
├── Uber.pbix                        # Power BI dashboard file
└── README.md
```