# -*- coding: utf-8 -*-

import time
import board
import adafruit_dht
import csv
from datetime import datetime
import serial

# Set up the DHT sensor (change to DHT22 if needed)
dht_device = adafruit_dht.DHT11(board.D4)

ser = serial.Serial('/dev/ttyACM0', 9600)  # Arduino serial

# CSV file to store data
csv_file = "sensor_data.csv"

# Write header if file doesn't exist
try:
    with open(csv_file, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Temperature (C)", "Humidity (%), 'Soil Moisture' "])
except FileExistsError:
    pass  # File already exists

# Logging loop
try:
    while True:
        try:
            temp = dht_device.temperature
            humidity = dht_device.humidity
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Read soil moisture from Arduino
            line = ser.readline().decode('utf-8').strip()

            print(f"{timestamp} | Temp: {temp}C | Humidity: {humidity}% | Moisture: {line}")

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, temp, humidity ,line])

        except RuntimeError as e:
            print("Sensor read error:", e)

        time.sleep(10)  # Log every 10 seconds

except KeyboardInterrupt:
    print("Logging stopped.")
