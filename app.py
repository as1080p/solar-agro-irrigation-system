from flask import Flask, render_template
import csv
import threading
import time
from joblib import load

model = load("irrigation_duration_model.pkl")

app = Flask(__name__)  # ? This line must come before @app.route

@app.route("/")
def dashboard():
    try:
        with open("sensor_data.csv", "r") as f:
            reader = list(csv.reader(f))
            data = reader[1:]  # Skip header
            data = data[-10:]  # Show last 10 entries
    except FileNotFoundError:
        data = []

    return render_template("dashboard.html", data=data, prediction=prediction_result)

prediction_result = "Unknown"

def run_prediction_loop():
    global prediction_result
    while True:
        try:
            with open("sensor_data.csv", "r") as f:
                reader = list(csv.reader(f))
                if len(reader) > 1:
                    latest = reader[-1]
                    # Extract features (adjust based on your model)
                    temp = float(latest[1])
                    humidity = float(latest[2])
                    moisture = float(latest[3])
                    features = [[temp, humidity, moisture]]
                    prediction = model.predict(features)[0]
                    prediction_result = f"Watering needed: {'Yes' if prediction else 'No'}"
        except Exception as e:
            prediction_result = f"Error: {e}"

        time.sleep(60)  # Wait 1 minute
    
if __name__ == "__main__":
    threading.Thread(target=run_prediction_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
