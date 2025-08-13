# 🌱 Smart Irrigation Dashboard

A real-time sensor dashboard built on Raspberry Pi 3 that monitors temperature, humidity, and soil moisture — and uses a machine learning model to predict irrigation needs. Designed for rapid prototyping, visual clarity, and presentation-ready polish.

---

## 📦 Project Overview

This project combines hardware sensors, live data logging, a Flask-based web dashboard, and an integrated ML model to create a smart irrigation assistant.

- 🧠 Predicts whether watering is needed using a trained DecisionTreeClassifier
- 🌡️ Logs DHT11 temperature & humidity readings
- 🌱 Reads soil moisture via Arduino ADC bridge
- 📊 Displays live data and trends in a responsive web interface
- 🧪 Updates predictions every minute in the background

---

## 🛠️ Hardware Setup

| Component            | Connection Details                     |
|---------------------|----------------------------------------|
| Raspberry Pi 3       | Hosts the dashboard and logger         |
| DHT11 Sensor         | GPIO4 (Data), 3.3V (VCC), GND          |
| Soil Moisture Sensor | A0 → Arduino → USB Serial to Pi        |

---

## 🧪 Software Stack

- **Python 3.11**
- **Flask** for web dashboard
- **Chart.js** for live sensor graphs
- **Bootstrap 5** for responsive UI
- **Adafruit_DHT** for sensor reading
- **pyserial** for Arduino communication
- **scikit-learn + joblib** for ML model integration

---

## 📁 Folder Structure

DHT_project/ ├── app.py                         # Flask dashboard ├── sensor_logger.py              # Logger script (DHT11 + soil moisture) ├── irrigation_duration_model.pkl # Trained ML model (DecisionTreeClassifier) ├── sensor_data.csv               # Live sensor log ├── templates/ │   └── dashboard.html            # Web UI template (Bootstrap + Chart.js)


---

## 🚀 How to Run

1. **Activate virtual environment**:
   ```bash
   source dht_env/bin/activate

2. **Launch the Dashboard**:
   ```bash
   python app.py
   
3. **View in Browser**: Open your browser and navigate to:
   ```bash
   http://<raspberry-pi-ip>:5000
Replace <raspberry-pi-ip> with your Raspberry Pi’s actual IP address.

4. **Activate virtual environment**:
   ```bash
   source dht_env/bin/activate

## 🤖 Machine Learning Integration

- Trained on historical sensor data
- Uses `[temperature, humidity, soil_moisture]` as input features
- Predicts binary output: **Watering needed → Yes / No**
- Executes every minute in a background thread for real-time updates

## 🎨 Dashboard Features

- Responsive layout powered by **Bootstrap**
- **Color-coded prediction badges** for quick visual feedback
- **Live sensor trends** rendered with **Chart.js**
- **Auto-refresh** every 10 seconds for up-to-date data
- Presentation-ready UI with custom **icons** and **branding**
