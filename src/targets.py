# src/targets.py
from .config import HORIZON, MOIST_COL, DURATION_TO_CLASS

def add_forecast_targets(df):
    for h in range(1, HORIZON+1):
        df[f'moisture_tplus{h}'] = df.groupby(['State Name','DistrictName'])[MOIST_COL].shift(-h)
    return df

def irrigation_rule_from_soil(row):
    # example rules based on dominant_soil and moisture (tune these later)
    m = row[MOIST_COL]
    s = row['dominant_soil']
    if pd.isna(m): 
        return 0
    if s == 'sandy':
        if m < 35: return 15
        elif m < 45: return 10
        elif m < 55: return 5
        else: return 0
    if s == 'floamy':
        if m < 25: return 15
        elif m < 35: return 10
        elif m < 45: return 5
        else: return 0
    if s == 'clayey':
        if m < 15: return 15
        elif m < 25: return 10
        elif m < 35: return 5
        else: return 0
    if s == 'clayskeletal':
        if m < 30: return 15
        elif m < 40: return 10
        elif m < 50: return 5
        else: return 0
    return 0

def add_irrigation_target(df):
    df['irrigation_duration'] = df.apply(irrigation_rule_from_soil, axis=1)
    df['irrigation_label'] = df['irrigation_duration'].map(DURATION_TO_CLASS)
    return df
