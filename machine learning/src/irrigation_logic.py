DESIRED_MOISTURE = 60
AREA_FACTOR = 1.0     # assumed
MOTOR_FLOW_RATE = 5  # liters per minute

def soil_absorption_factor(clay, loam, sand):
    return (0.7 * clay) + (0.5 * loam) + (0.3 * sand)

def irrigation_duration(predicted_moisture, clay, loam, sand, rain_prob):
    deficit = DESIRED_MOISTURE - predicted_moisture

    if deficit <= 0 or rain_prob > 0.4:
        return 0

    saf = soil_absorption_factor(clay, loam, sand)
    water_needed = deficit * saf * AREA_FACTOR
    time_minutes = water_needed / MOTOR_FLOW_RATE

    if time_minutes < 1:
        return 1
    elif time_minutes < 3:
        return 3
    else:
        return 5

