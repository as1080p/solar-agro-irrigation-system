# src/inference.py
import joblib
import pandas as pd
from .config import PROCESSED_CSV, STATE_COL, CLASS_TO_DURATION


# feature lists
FORECAST_FEATS = [
    'Aggregate Soilmoisture Percentage (at 15cm)',
    'Aggregate Soilmoisture Percentage (at 15cm)_lag_1',
    'Aggregate Soilmoisture Percentage (at 15cm)_lag_3',
    'Aggregate Soilmoisture Percentage (at 15cm)_lag_7',
    'Aggregate Soilmoisture Percentage (at 15cm)_roll7',
    'month_sin','month_cos',
    'sandy','floamy','clayey','clayskeletal','drainage_score'
]

CLF_FEATS = [
    'Aggregate Soilmoisture Percentage (at 15cm)',
    'moisture_tplus1',
    'sandy','floamy','clayey','clayskeletal',
    'drainage_score','month_sin','month_cos'
]

def generate_preds():
    # Load processed data and filter for Maharashtra
    df_full = pd.read_csv(PROCESSED_CSV)
    df = df_full[df_full[STATE_COL].str.strip().str.lower() == 'maharashtra'].copy()
    forecast_pipe = joblib.load('C:\\Users\\subha\\OneDrive\\Documents\\GitHub\\solar-agro-irrigation-system\\models\\moisture_forecast_maharashtra.pkl')
    X = df[FORECAST_FEATS].fillna(0)
    preds = forecast_pipe.predict(X)
    # Add forecast columns with correct names
    for i in range(1, 8):
        df[f'moisture_tplus{i}'] = preds[:, i-1]
    df.to_csv('data/processed/fe_with_preds_maharashtra.csv', index=False)
    print("Saved data/processed/fe_with_preds_maharashtra.csv")

def classify_and_output():
    df = pd.read_csv('data/processed/fe_with_preds_maharashtra.csv')
    clf = joblib.load('models/irrigation_rf.pkl')
    X = df[CLF_FEATS].fillna(0)
    preds = clf.predict(X)
    df['pred_class'] = preds  # assign predicted classes here
    df['pred_duration_sec'] = df['pred_class'].map(CLASS_TO_DURATION)
    df.to_csv('data/processed/predictions_output_maharashtra.csv', index=False)
    print("Saved data/processed/predictions_output_maharashtra.csv")

if __name__ == "__main__":
    generate_preds()
    classify_and_output()
    print("✅ Inference complete!")
