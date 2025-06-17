FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .

RUN pip install --no-cache-dir pandas scikit-learn pyarrow
ENTRYPOINT ["python", "starter.py"]
