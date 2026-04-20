# model_pipeline.py
from src.data_loader import load_raw
from src.features import add_time_features, soil_features, create_lag_roll
from src.targets import add_forecast_targets, add_irrigation_target
import pandas as pd

df = load_raw()
df = add_time_features(df)
df = soil_features(df)
df = create_lag_roll(df)
df = add_forecast_targets(df)
df = add_irrigation_target(df)
# drop rows with NA target (end of series)
df_out = df.dropna(subset=[f'moisture_tplus{i}' for i in range(1,8)])
df_out.to_csv("data/processed/processed_dataset_Created.csv", index=False)
print("Processed dataset saved.")
