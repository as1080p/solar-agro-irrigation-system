import pandas as pd
import numpy as np

DATA_PATH = "../data/raw/Combined_Dataset_final.csv"

df = pd.read_csv(DATA_PATH)
print(df.head())

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

soil_cols = ["clayey", "floamy", "sandy", "clayskeletal"]

soil_profile = (
    df.groupby("districtname")[soil_cols]
    .mean()
    .reset_index()
)

soil_profile.to_csv("../data/processed/soil_profile.csv", index=False)

df = df[
    [
        "date",
        "statename",
        "districtname",
        "volume_soilmoisture_percentage_at_15cm"
    ]
]

df["date"] = pd.to_datetime(df["date"], dayfirst=True)


