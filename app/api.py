import joblib
import pandas as pd
from fastapi import FastAPI


app = FastAPI()
model = joblib.load('models/best_pipeline.joblib')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(student: dict):
    data = pd.DataFrame([student], columns=model.feature_names_in_)
    data = data.fillna(float('nan'))
    prediction = model.predict(data)[0]
    return {'prediction': str(prediction)}
