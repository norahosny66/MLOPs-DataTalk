# integration_test.py

import pandas as pd
from datetime import datetime
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

year = 2023
month = 1

input_file = f's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'

s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')

options = {
    'client_kwargs': {
        'endpoint_url': s3_endpoint_url
    }
}

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

import os

# Set environment variables
os.environ['S3_ENDPOINT_URL'] = 'http://localhost:4566'
os.environ['INPUT_FILE_PATTERN'] = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
os.environ['OUTPUT_FILE_PATTERN'] = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'

# Run the batch script
os.system('python batch.py 2023 1')

output_file = 's3://nyc-duration/out/2023-01.parquet'

options = {
    'client_kwargs': {
        'endpoint_url': 'http://localhost:4566'
    }
}

df_result = pd.read_parquet(output_file, storage_options=options)
print("Sum of predicted durations:", df_result['predicted_duration'].sum())
