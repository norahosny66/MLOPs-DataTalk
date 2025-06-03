# 🚖 NYC Taxi Trip Duration Prediction

This project trains a linear regression model to predict taxi trip durations in NYC using 2023 trip data. The workflow is orchestrated using Mage.ai, and the model is tracked and logged using MLflow.

---

## 📊 Dataset

- **Source:** NYC Yellow Taxi Trip Records  
- **File:** `yellow_tripdata_2023-03.parquet`  
- **Features used:** `PULocationID`, `DOLocationID`, `duration`

---

## ⚙️ Steps

### Step-by-step Guide

1. **Prepare your data**  
   Download the NYC Yellow Taxi dataset (Parquet format).  
   Place the file inside your Mage project directory, e.g.:  


2. **Set up MLflow tracking server**  
Run MLflow tracking server on your VM (outside Mage container), for example:  
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
Configure Mage container
Since Mage runs inside a Docker container, install MLflow inside the Mage container:

bash
Copy
Edit
docker exec -it <mage_container_name> /bin/bash
pip install mlflow
In your code inside Mage, configure MLflow tracking URI with your VM IP, e.g.:

python
Copy
Edit
import mlflow
mlflow.set_tracking_uri("http://192.168.x.x:5000")
Write your Mage blocks

Data loader block: loads and preprocesses the data.

Transformer block: trains the model and logs it to MLflow.

Make sure to use decorators correctly:

python
Copy
Edit
from mage_ai.data_preparation.decorators import data_loader, transformer

@data_loader
def load_data():
    # Load parquet file and preprocess

@transformer
def train_model(df):
    # Train model, log metrics and artifacts with MLflow
Run and monitor
Start Mage project:

bash
Copy
Edit
mage start my_first_project
Access MLflow UI from your browser:

cpp
Copy
Edit
http://192.168.x.x:5000
View your experiments, runs, metrics, and logged model artifacts.

Verify model size
The model is saved inside the MLflow artifacts directory.
Model size: 4500 bytes (found in model/MLmodel under model_size_bytes).

📈 Results
Feature Matrix Shape: (3316216, 518)

Train RMSE: ~66.56

Model Intercept: ~24.77

🔍 MLflow UI
Access URL: http://<mlflow-host>:5000
