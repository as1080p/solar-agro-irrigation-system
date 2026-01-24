def simulate_analog(base_value):
    noise = np.random.normal(0, 2)
    drift = np.random.uniform(-1, 1)
    return base_value + noise + drift

def analog_to_moisture(val):
    if val >= 226:
        return 0
    elif 148 <= val < 226:
        return (226 - val) * (60 / (226 - 148))
    elif 173 <= val < 148:
        return 60 + (148 - val) * (20 / (148 - 173))
    else:
        return min(100, 80 + (188 - val) * (20 / (188 - 173)))

def generate_sensor_reading(moisture):
    analog = simulate_analog(226 - moisture)
    temp = np.random.uniform(22, 35)
    humidity = np.random.uniform(40, 90)
    return analog, temp, humidity

