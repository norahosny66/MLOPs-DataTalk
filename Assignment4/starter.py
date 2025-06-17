#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pickle
import pandas as pd
import sys
import sklearn

print("Scikit-learn version:", sklearn.__version__)


# In[2]:

print("Python version:", sys.version)

# In[22]:





# In[23]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[24]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[25]:


year = int(sys.argv[1])
month = int(sys.argv[2])
df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


# In[8]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[10]:


import numpy as np

std_pred = np.std(y_pred)
print(f"Standard deviation of predicted durations: {std_pred}")


# In[12]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[15]:


df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicted_duration': y_pred
})
df_result.to_parquet(
    'output_file',
    engine='pyarrow',
    compression=None,
    index=False
)


# In[17]:


import os

file_size = os.path.getsize('output_file')
print(f"File size: {file_size / 1024:.2f} KB")
print("Mean predicted duration:", y_pred.mean())

# In[ ]:




