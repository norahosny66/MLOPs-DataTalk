import os
import pickle
import click
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)



def run_train(data_path: str):
    print("Loading training data...")
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    print("Loading validation data...")
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    sample_size = 10000
    X_train = X_train[:sample_size]
    y_train = y_train[:sample_size]
    mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.sklearn.autolog(log_datasets=False,log_input_examples=False,log_model_signatures=False)


    with mlflow.start_run():  # Track this run
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        print("Starting MLflow run...")
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        print(f"RMSE: {rmse}")
        
        print("min_samples_split:", rf.min_samples_split)

if __name__ == '__main__':
    run_train()
