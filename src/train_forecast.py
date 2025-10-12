# src/train_forecast.py
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.config import PROCESSED_CSV
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
    df = pd.read_csv(PROCESSED_CSV)
    # drop NA rows created by lags or targets
    df = df.dropna(subset=FEATURES + [f'moisture_tplus{i}' for i in range(1,8)])
    X = df[FEATURES]
    y = df[[f'moisture_tplus{i}' for i in range(1,8)]]
    pipe = Pipeline([('scaler', StandardScaler()), ('svr', MultiOutputRegressor(SVR(kernel='rbf')))])
    pipe.fit(X, y)
    joblib.dump(pipe, 'models/moisture_forecast.pkl')
    print("Saved models/moisture_forecast.pkl")

if __name__ == "__main__":
    train()
