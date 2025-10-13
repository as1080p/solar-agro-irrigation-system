# src/train_forecast.py
import joblib
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import PROCESSED_CSV, STATE_COL  # assuming STATE_COL = 'State'

FEATURES = [
    'Aggregate Soilmoisture Percentage (at 15cm)',
    'Aggregate Soilmoisture Percentage (at 15cm)_lag_1',
    'Aggregate Soilmoisture Percentage (at 15cm)_lag_3',
    'Aggregate Soilmoisture Percentage (at 15cm)_lag_7',
    'Aggregate Soilmoisture Percentage (at 15cm)_roll7',
    'month_sin','month_cos',
    'sandy','floamy','clayey','clayskeletal','drainage_score'
]

def train():
    print(f"Reading from: {PROCESSED_CSV}")
    df = pd.read_csv(PROCESSED_CSV)

    # Filter for Maharashtra only and take first 200 rows
    df = df[df[STATE_COL].str.lower() == 'maharashtra'].head(10170)
    print(f"Filtered dataset size: {len(df)} rows")

    # Drop NA rows (due to lags)
    df = df.dropna(subset=FEATURES + [f'moisture_tplus{i}' for i in range(1,8)])

    X = df[FEATURES]
    y = df[[f'moisture_tplus{i}' for i in range(1,8)]]

    print("Training model on Maharashtra data only...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', MultiOutputRegressor(SVR(kernel='rbf')))
    ])
    pipe.fit(X, y)

    joblib.dump(pipe, 'models/moisture_forecast_maharashtra.pkl')
    print("✅ Saved model: models/moisture_forecast_maharashtra.pkl")

if __name__ == "__main__":
    train()
