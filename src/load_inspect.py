import pandas as pd
df = pd.read_csv("data/Combined_Dataset_final.csv")  # your file
print(df.columns.tolist())
print(df.head())
print(df.info())
# Check missing
print(df.isna().sum())