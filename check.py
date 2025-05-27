import pandas as pd

df = pd.read_parquet('./home/yellow_tripdata_2025-01.parquet')  # or .csv
print(df.columns)