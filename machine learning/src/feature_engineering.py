soil_profile = pd.read_csv("../data/processed/soil_profile.csv")

df = df.merge(
    soil_profile,
    on="districtname",
    how="left"
)

df = df.sort_values("date")

df["day_of_year"] = df["date"].dt.dayofyear

df["moisture_trend"] = (
    df["volume_soilmoisture_percentage_at_15cm"]
    .diff()
    .fillna(0)
)

FEATURES = [
    "volume_soilmoisture_percentage_at_15cm",
    "moisture_trend",
    "clayey",
    "floamy",
    "sandy",
    "day_of_year"
]

df["target_moisture"] = (
    df["volume_soilmoisture_percentage_at_15cm"]
    .shift(-6)
)

df = df.dropna()

df.to_csv("../data/processed/training_data.csv", index=False)

