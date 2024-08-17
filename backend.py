from fastapi import FastAPI, HTTPException, File, UploadFile
from io import StringIO
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import load_model

app = FastAPI()

# URL of the model file
MODEL_URL = "https://github.com/Pricile21/E-NOSE-NASA-COVID_19-Pr-diction/raw/master/best_model.keras"
MODEL_PATH = "best_model.keras"

# Download the model file from the URL
response = requests.get(MODEL_URL)
if response.status_code == 200:
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
else:
    raise Exception("Failed to download the model file.")
model = load_model(MODEL_PATH)

# Initialiser le scaler globalement
scaler = StandardScaler()

def preprocess_data(data, is_training=True):
    if 'IndividualID' in data.columns:
        data.rename(columns={'IndividualID': 'ID'}, inplace=True)
    data['Min:Sec'] = data['Min:Sec'].apply(lambda x: int(x.split(':')[0]) * 60 + float(x.split(':')[1]))
    data = data.groupby('ID').mean().reset_index()
    if is_training:
        X = data.drop(columns=['Result'])
        y = data['Result']
        scaler.fit(X)
        return X, y
    else:
        X = data
        return X

TRAIN_DATA_URL = "https://github.com/Pricile21/E-NOSE-NASA-COVID_19-Pr-diction/raw/master/train_data.csv"

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API FastAPI pour le modèle E-NOSE"}

@app.get("/variables")
async def list_variables():
    try:
        response = requests.get(TRAIN_DATA_URL)
        if response.status_code == 200:
            csv_content = StringIO(response.text)
            train_data = pd.read_csv(csv_content)
            variables = train_data.columns.tolist()
            return {"variables": variables}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to download the train data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    try:
        file_location = "temp_train_data.csv"
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        train_data = pd.read_csv(file_location)
        print(train_data.head())
        X_train, y_train = preprocess_data(train_data, is_training=True)
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_split.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_ckp = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor="val_accuracy", mode="max", save_best_only=True)
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=200, restore_best_weights=True)
        model.fit(X_train_split, y_train_split, epochs=200, batch_size=2048, validation_data=(X_val_split, y_val_split), callbacks=[model_ckp, stop])
        model.save(MODEL_PATH)
        return {"message": "Modèle entraîné avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def make_prediction(file: UploadFile = File(...)):
    try:
        file_location = "temp_test_data.csv"
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        test_data = pd.read_csv(file_location)
        X_test = preprocess_data(test_data, is_training=False)
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        binary_predictions = (predictions > 0.5).astype(int)
        result_predictions = ["Positive" if pred == 1 else "Negative" for pred in binary_predictions]
        return {"predictions": result_predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8019)
