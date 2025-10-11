# src/clean.py
import pandas as pd
def clean(df):
    # Strip column names
    df.columns = [c.strip() for c in df.columns]
    # Parse date (adjust format if needed; sample '31-01-2020')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    # Lowercase names
    df['State Name'] = df['State Name'].astype(str).str.strip()
    df['DistrictName'] = df['DistrictName'].astype(str).str.strip()
    # Convert numeric columns
    num_cols = ['Average Soilmoisture Level (at 15cm)',
                'Average SoilMoisture Volume (at 15cm)',
                'Aggregate Soilmoisture Percentage (at 15cm)',
                'Volume Soilmoisture percentage (at 15cm)',
                'clayey','floamy','sandy','clayskeletal']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop rows with missing date or district
    df = df.dropna(subset=['Date','DistrictName'])
    return df

# usage
# df = clean(df)
