def rain_probability(temp_now, temp_prev, humidity):
    temp_drop = temp_prev - temp_now

    if humidity > 80 and temp_drop > 2:
        return 0.7
    elif humidity > 70:
        return 0.4
    else:
        return 0.1

