if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

@transformer
def transform(df, *args, **kwargs):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    print(f'Feature matrix size: {X_train.shape}')

    y_train = df['duration'].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)
    rmse = mean_squared_error(y_train, y_pred)
    #mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_tracking_uri('http://172.17.0.1:5000')
    print(f'Train RMSE: {rmse:.4f}')
    print(f'Intercept: {lr.intercept_:.2f}')
    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "model")
        mlflow.log_metric("rmse", rmse)
    return dv, lr

