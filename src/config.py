# src/config.py
RAW_CSV = "data/raw/Combined_Dataset_final.csv"
PROCESSED_CSV = "data/processed/prcesses_dataset.csv"
DATE_COL = "Date"
STATE_COL = "State Name"
DIST_COL = "DistrictName"
MOIST_COL = "Aggregate Soilmoisture Percentage (at 15cm)"

SOIL_FRAC_COLS = ["clayey","floamy","sandy","clayskeletal"]
HORIZON = 7

# irrigation classes mapping
DURATION_TO_CLASS = {0:0, 5:1, 10:2, 15:3}
CLASS_TO_DURATION = {v:k for k,v in DURATION_TO_CLASS.items()}