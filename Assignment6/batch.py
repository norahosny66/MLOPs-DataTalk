import os
import pandas as pd
import pickle

#  Helper functions
def get_input_path(year, month):
    default_input = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    pattern = os.getenv('INPUT_FILE_PATTERN', default_input)
    return pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output)
    return pattern.format(year=year, month=month)

#  Your read_data and prepare_data
def read_data(path, categorical):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    
    if s3_endpoint_url:
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }
        df = pd.read_parquet(path, storage_options=options)
    else:
        df = pd.read_parquet(path)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df
def save_data(df, output_file):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    options = {}

    if s3_endpoint_url:
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }

    df.to_parquet(
        output_file,
        engine='pyarrow',
        index=False,
        storage_options=options
    )

#  Main logic
def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    categorical = ['PULocationID', 'DOLocationID']

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    save_options = {
        'engine': 'pyarrow',
        'index': False
    }

    if s3_endpoint_url:
        save_options['storage_options'] = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }

    df_result.to_parquet(output_file, **save_options)

# Entry point
if __name__ == '__main__':
    import sys
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
