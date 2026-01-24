import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("../data/processed/training_data.csv")

X = df[
    [
        "volume_soilmoisture_percentage_at_15cm",
        "moisture_trend",
        "clayey",
        "floamy",
        "sandy",
        "day_of_year"
    ]
]

y = df["target_moisture"]

split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)

print("MAE:", mae)
print("RMSE:", rmse)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Soil Moisture (%)")
plt.title("Soil Moisture Prediction")
plt.savefig("../outputs/plots/soil_moisture_prediction.png", dpi=300)
plt.show()

joblib.dump(model, "../models/soil_moisture_predictor.pkl")

