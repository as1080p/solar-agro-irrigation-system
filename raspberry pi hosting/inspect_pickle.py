import pickletools

with open("machine learning\models\irrigation_rf.pkl", "rb") as f:
    raw = f.read()
    print(pickletools.dis(raw[:100]))  # Just a peek at the first 100 bytes
