import pickletools

with open("irrigation_duration_model.pkl", "rb") as f:
    raw = f.read()
    print(pickletools.dis(raw[:100]))  # Just a peek at the first 100 bytes
