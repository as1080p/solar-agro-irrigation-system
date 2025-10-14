# src/train_classifier.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .config import PROCESSED_CSV

CLF_FEATS = [
    'Aggregate Soilmoisture Percentage (at 15cm)',
    'moisture_tplus1',   # must be generated before training
    'sandy', 'floamy', 'clayey', 'clayskeletal',
    'drainage_score', 'month_sin', 'month_cos'
]

def train():
    print(f"Reading from: {PROCESSED_CSV}", flush=True)
    df = pd.read_csv(PROCESSED_CSV)

    # ✅ Filter for Maharashtra only
    df = df[df['State Name'].str.lower() == 'maharashtra']

    print(f"Filtered dataset shape (Maharashtra only): {df.shape}")

    # Drop rows missing required columns
    df = df.dropna(subset=CLF_FEATS + ['irrigation_label'])

    X = df[CLF_FEATS]
    y = df['irrigation_label']

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])

    print("Training classifier...", flush=True)
    pipeline.fit(X, y)
    joblib.dump(pipeline, 'models/irrigation_rf.pkl')
    print("✅ Saved Maharashtra-only model to models/irrigation_rf.pkl")

if __name__ == "__main__":
    train()
