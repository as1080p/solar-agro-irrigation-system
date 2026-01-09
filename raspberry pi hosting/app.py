from flask import Flask, request, jsonify, render_template
import pandas as pd, os
from datetime import datetime
import joblib

app = Flask(__name__)
CSV_FILE = "sensor_log.csv"

# Load your pre-trained model (once you’ve trained it)
# model = joblib.load("irrigation_rf.pkl")

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    moisture = float(data['moisture'])
    temp = float(data['temperature'])
    hum = float(data['humidity'])

    # append to CSV
    row = pd.DataFrame([[datetime.now(), moisture, temp, hum]],
                       columns=['Timestamp','Moisture(%)','Temperature','Humidity'])
    row.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)

    # --- Rule-based logic prediction ---
    watering = "No"
    if hum > 80:
        watering = "No (High humidity)"
    elif moisture < 30:
        watering = "Yes (Dry soil)"
    elif moisture > 45:
        watering = "No (Soil moist)"

    # --- If ML model available ---
    # pred = model.predict([[moisture,temp,hum]])[0]
    # watering = "Yes" if pred==1 else "No"

    return jsonify({"status":"logged","watering":watering}),200

@app.route('/')
def dashboard():
    if not os.path.exists(CSV_FILE):
        return "No data yet!"
    df = pd.read_csv(CSV_FILE).tail(10)
    return render_template('dashboard.html', tables=[df.to_html(index=False)], titles=df.columns.values)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
