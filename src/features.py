import numpy as np
import pandas as pd

def add_time_features(df):
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    # cyclical months
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    return df

def soil_dominant_and_features(df):
    # You already have four fractional columns per row: clayey,floamy,sandy,clayskeletal
    soil_cols = ['clayey','floamy','sandy','clayskeletal']
    # If they are proportions, keep them. Add a 'dominant' column via argmax
    df['dominant_soil'] = df[soil_cols].idxmax(axis=1)
    # Optionally create a numeric 'drainage_score' as weighted sum (higher value => faster drainage)
    weights = {'sandy':1.2, 'floamy':1.0, 'clayey':0.7, 'clayskeletal':1.0}
    df['drainage_score'] = (
        df['sandy']*weights['sandy'] +
        df['floamy']*weights['floamy'] +
        df['clayey']*weights['clayey'] +
        df['clayskeletal']*weights['clayskeletal']
    )
    return df

def create_lag_features(df):
    group_cols = ['State Name', 'DistrictName']
    target_col = 'Aggregate Soilmoisture Percentage (at 15cm)'
    df = df.sort_values(group_cols + ['Date'])
    for l in [1,2,3,7]:
        df[f'{target_col}_lag_{l}'] = df.groupby(group_cols)[target_col].shift(l)
    df[f'{target_col}_roll7'] = (
        df.groupby(group_cols)[target_col]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)  # <-- fix: drop both group levels
    )
    # after making lags you will have NaNs for first rows per group; keep or drop depending on approach
    return df