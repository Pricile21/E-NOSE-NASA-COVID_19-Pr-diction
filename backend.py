from fastapi import FastAPI, HTTPException, File, UploadFile
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import load_model

app = FastAPI()

# URL of the model file
MODEL_URL = "https://github.com/Pricile21/E-NOSE-NASA-COVID_19-Pr-diction/raw/master/best_model.keras"

# Local path to save the downloaded model file
MODEL_PATH = "best_model.keras"

# Download the model file from the URL
response = requests.get(MODEL_URL)
if response.status_code == 200:
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
else:
    raise Exception("Failed to download the model file.")

# Load the model
model = load_model(MODEL_PATH)

# Initialiser le scaler globalement
scaler = StandardScaler()

# Charger et prétraiter les données d'entraînement
def preprocess_data(data, is_training=True):
    # Renommer la colonne 'IndividualID' en 'ID' si elle existe
    if 'IndividualID' in data.columns:
        data.rename(columns={'IndividualID': 'ID'}, inplace=True)
    
    # Conversion de la colonne 'Min:Sec' en secondes
    data['Min:Sec'] = data['Min:Sec'].apply(lambda x: int(x.split(':')[0]) * 60 + float(x.split(':')[1]))
    
    # Agrégation des données par 'ID' en prenant la moyenne pour chaque groupe
    data = data.groupby('ID').mean().reset_index()

    if is_training:
        # Ajuster le scaler uniquement avec les données d'entraînement
        X = data.drop(columns=['Result'])
        y = data['Result']
        scaler.fit(X)
        return X, y
    else:
        # Transformer les données de test avec le scaler ajusté sur les données d'entraînement
        X = data
        return X
# URL of the train data CSV file
TRAIN_DATA_URL = "https://github.com/Pricile21/E-NOSE-NASA-COVID_19-Pr-diction/raw/master/train_data.csv"

# Endpoint pour obtenir les variables du jeu de données
@app.get("/variables")
async def list_variables():
    try:
        # Download the train data CSV file from the URL
        response = requests.get(TRAIN_DATA_URL)
        if response.status_code == 200:
            # Read the CSV content into a pandas DataFrame
            csv_content = StringIO(response.text)
            train_data = pd.read_csv(csv_content)
            
            # Extract the column names
            variables = train_data.columns.tolist()
            return {"variables": variables}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to download the train data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour entraîner le modèle avec un fichier
@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    try:
        # Sauvegarder le fichier téléchargé
        file_location = "temp_train_data.csv"
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        # Charger les données d'entraînement
        train_data = pd.read_csv(file_location)
        print(train_data.head())

        # Prétraiter les données
        X_train, y_train = preprocess_data(train_data, is_training=True)

        # Normalisation des données
        X_train_scaled = scaler.fit_transform(X_train)

        # Split the training data into training and validation sets
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

        # Définir le modèle
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_split.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compilation du modèle
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Définir les callbacks
        model_ckp = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH,
                                                       monitor="val_accuracy",
                                                       mode="max",
                                                       save_best_only=True)
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=200, restore_best_weights=True)

        # Entraînement du modèle
        model.fit(X_train_split, y_train_split, epochs=200, batch_size=2048,
                  validation_data=(X_val_split, y_val_split),
                  callbacks=[model_ckp, stop])

        # Sauvegarder le modèle
        model.save(MODEL_PATH)

        return {"message": "Modèle entraîné avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour faire des prédictions avec un fichier
@app.post("/predict")
async def make_prediction(file: UploadFile = File(...)):
    try:
        # Sauvegarder le fichier téléchargé
        file_location = "temp_test_data.csv"
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        # Charger les données de test
        test_data = pd.read_csv(file_location)
        
        # Prétraiter les données
        X_test = preprocess_data(test_data, is_training=False)
        
        # Normalisation des données
        X_test_scaled = scaler.transform(X_test)

        # Faire des prédictions
        predictions = model.predict(X_test_scaled)
        binary_predictions = (predictions > 0.5).astype(int)

        # Convertir les prédictions en "Positive" ou "Negative"
        result_predictions = ["Positive" if pred == 1 else "Negative" for pred in binary_predictions]

        return {"predictions": result_predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8015)  # Utiliser localhost pour éviter les problèmes de réseau
