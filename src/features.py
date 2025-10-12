# src/features.py
import numpy as np
import pandas as pd
from src.config import SOIL_FRAC_COLS, STATE_COL, DIST_COL, DATE_COL, MOIST_COL

def add_time_features(df):
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df['month'] = df[DATE_COL].dt.month
    df['day'] = df[DATE_COL].dt.day
    df['dayofyear'] = df[DATE_COL].dt.dayofyear
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    
    return df

def soil_features(df):
    # dominant soil (string)
    df['dominant_soil'] = df[SOIL_FRAC_COLS].idxmax(axis=1)
    # drainage score — example weighted sum
    weights = {'sandy':1.2, 'floamy':1.0, 'clayey':0.8, 'clayskeletal':0.7}
    df['drainage_score'] = sum(df[col] * weights[col] for col in SOIL_FRAC_COLS)
    return df

def create_lag_roll(df):
    from src.config import MOIST_COL, STATE_COL, DIST_COL

    # 🧹 Ensure moisture column is numeric
    df[MOIST_COL] = pd.to_numeric(df[MOIST_COL], errors='coerce')
    df[MOIST_COL].fillna(df[MOIST_COL].median(), inplace=True)

    # ✅ Then proceed with rolling calculations
    df[f'{MOIST_COL}_lag1'] = df.groupby([STATE_COL, DIST_COL])[MOIST_COL].shift(1)
    df[f'{MOIST_COL}_roll7'] = (
        df.groupby([STATE_COL, DIST_COL])[MOIST_COL]
          .rolling(7, min_periods=1)
          .mean()
          .reset_index(0, drop=True)
    )
    return df
