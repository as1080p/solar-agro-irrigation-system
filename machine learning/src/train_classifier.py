# train_duration_model.py
"""
Train a RandomForest classifier to predict irrigation durations in seconds {0,5,10,15}
using your Maharashtra-only dataset. Synthetic labels are created from
'Aggregate Soilmoisture Percentage (at 15cm)' with optional temp/humidity adjustment.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Paths
RAW_PROCESSED = "data/processed/processed_dataset_Created.csv"  # or your processed CSV path
OUT_MODEL = "models/irrigation_duration_rf.pkl"
OUT_ENCODER = "models/district_encoder.joblib"

# Config: discrete durations (seconds)
DURATIONS = [0, 5, 10, 15]  # target classes

# Synthetic label thresholds (based on Aggregate Soilmoisture Percentage)
# You said short durations are fine — thresholds tuned to Maharashtra patterns later.
def synthetic_duration_from_moisture(m_pct, temp=None, hum=None):
    """
    Map moisture% -> duration seconds with small adjustments for high temp/low humidity.
    Lower moisture -> higher duration.
    """
    if pd.isna(m_pct):
        return 0

    # Base mapping: higher moisture -> lower duration
    if m_pct > 60:
        base = 0
    elif m_pct > 50:
        base = 5
    elif m_pct > 40:
        base = 10
    else:
        base = 15

    # Weather adjustment: if it's hot and dry, increase duration one step
    if temp is not None and hum is not None:
        if (temp >= 30 and hum <= 40) or (temp >= 33):
            # bump one step towards more water
            idx = DURATIONS.index(base)
            idx = min(len(DURATIONS)-1, idx + 1)
            base = DURATIONS[idx]
    return base

def main():
    os.makedirs("models", exist_ok=True)
    print("Loading processed dataset:", RAW_PROCESSED)
    df = pd.read_csv(RAW_PROCESSED)

    # Keep Maharashtra only
    df = df[df['State Name'].str.strip().str.lower() == 'maharashtra'].copy()
    print("Filtered rows (Maharashtra):", len(df))

    # Make sure the main moisture column exists
    MOIST_COL = 'Aggregate Soilmoisture Percentage (at 15cm)'
    if MOIST_COL not in df.columns:
        raise RuntimeError(f"{MOIST_COL} not in dataset")

    # Optional DHT11 columns: ensure present (if not, they will be NaN)
    TEMP_COL = 'Temperature'
    HUM_COL = 'Humidity'

    # Create month cyc features (helps)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['month'] = df['Date'].dt.month.fillna(0).astype(int)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)

    # Create synthetic target
    df['irrigation_duration_sec'] = df.apply(
        lambda r: synthetic_duration_from_moisture(
            r.get(MOIST_COL, np.nan),
            r.get(TEMP_COL, np.nan),
            r.get(HUM_COL, np.nan)
        ), axis=1
    )

    # Filter/remove rows without soil fractions (optional: fill 0)
    soil_cols = ['clayey','floamy','sandy','clayskeletal']
    for c in soil_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[soil_cols] = df[soil_cols].fillna(0.0)

    # Features
    FEATURES = [
        MOIST_COL,
        'month_sin','month_cos',
        'Average Soilmoisture Level (at 15cm)',
        'Average SoilMoisture Volume (at 15cm)',
        'Volume Soilmoisture percentage (at 15cm)',
        'clayey','floamy','sandy','clayskeletal',
        TEMP_COL, HUM_COL
    ]

    # Keep only relevant cols
    df_feats = df[FEATURES + ['DistrictName', 'irrigation_duration_sec']].copy()

    # Drop rows where target missing (shouldn't happen)
    df_feats = df_feats.dropna(subset=['irrigation_duration_sec'])

    # Encode district names (LabelEncoder)
    le = LabelEncoder()
    df_feats['district_enc'] = le.fit_transform(df_feats['DistrictName'].astype(str))
    joblib.dump(le, OUT_ENCODER)
    print("Saved district encoder:", OUT_ENCODER)

    # Final feature list
    X = df_feats[FEATURES + ['district_enc']].values
    y = df_feats['irrigation_duration_sec'].values

    # Train/test split (time-aware would be ideal; here simple split as baseline)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Simple pipeline with imputation + RF
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_imp, y_train)
    y_pred = clf.predict(X_test_imp)

    print("Classification report (test):")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the pipeline pieces
    joblib.dump({
        'model': clf,
        'imputer': imputer,
        'features': FEATURES + ['district_enc'],
        'district_encoder': le
    }, OUT_MODEL)
    print("Saved model bundle to:", OUT_MODEL)

if __name__ == "__main__":
    main()
