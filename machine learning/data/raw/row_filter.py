import pandas as pd

# Load the CSV file (adjust the path as needed)
df = pd.read_csv("Combined_Dataset_final.csv")

# Filter rows where 'State Name' is 'Maharashtra'
maharashtra_df = df[df['State Name'] == 'Maharashtra']

# Save the filtered data to a new CSV file
maharashtra_df.to_csv("Combined_Dataset_Maharashtra.csv", index=False)