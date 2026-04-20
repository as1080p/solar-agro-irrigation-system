# src/data_loader.py
import pandas as pd
from .config import RAW_CSV, DATE_COL

def load_raw(path=RAW_CSV):
    df = pd.read_csv(path)
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    # parse date (sample inline format dd-mm-yyyy)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors='coerce')
    return df