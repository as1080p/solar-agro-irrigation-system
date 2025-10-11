import pandas as pd
from src.features import add_time_features, create_lag_features, soil_dominant_and_features

df = pd.read_csv("data/Combined_Dataset_final.csv")

# Handle missing values (example: fill with 0 or use another strategy)
df = df.fillna(0)

# Convert 'Aggregate Soilmoisture Percentage (at 15cm)' to numeric
df['Aggregate Soilmoisture Percentage (at 15cm)'] = pd.to_numeric(
    df['Aggregate Soilmoisture Percentage (at 15cm)'], errors='coerce'
)

# Convert 'Date' column to datetime, coerce errors, and drop invalid rows
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date'])

df = add_time_features(df)
df = soil_dominant_and_features(df)
df = create_lag_features(df)

# Forecast targets
horizon = 7
for h in range(1, horizon+1):
    df[f'moisture_tplus{h}'] = df.groupby(['State Name','DistrictName'])['Aggregate Soilmoisture Percentage (at 15cm)'].shift(-h)
df_forecast = df.dropna(subset=[f'moisture_tplus{h}' for h in range(1, horizon+1)]).copy()

def duration_from_soil_moisture(row):
    m = row['Aggregate Soilmoisture Percentage (at 15cm)']
    s = row['dominant_soil']
    if pd.isna(m): return 0
    if s == 'sandy':
        if m < 35: return 15
        elif m < 45: return 10
        elif m < 55: return 5
        else: return 0
    elif s == 'floamy':
        if m < 25: return 15
        elif m < 35: return 10
        elif m < 45: return 5
        else: return 0
    elif s == 'clayey':
        if m < 15: return 15
        elif m < 25: return 10
        elif m < 35: return 5
        else: return 0
    elif s == 'clayskeletal':
        if m < 30: return 15
        elif m < 40: return 10
        elif m < 50: return 5
        else: return 0
    else:
        # default
        if m < 20: return 15
        elif m < 30: return 10
        elif m < 40: return 5
        else: return 0

df_forecast['irrigation_duration'] = df_forecast.apply(duration_from_soil_moisture, axis=1)
# Map duration to class index for classifier:
class_map = {0:0, 5:1, 10:2, 15:3}
df_forecast['irrigation_label'] = df_forecast['irrigation_duration'].map(class_map)
