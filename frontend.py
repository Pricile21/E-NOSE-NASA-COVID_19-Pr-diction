import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(
    page_title="Prédiction 📈 du diagnostic de l’haleine",
    page_icon=":chart_with_upwards_trend:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Prédiction 📈 du diagnostic de l’haleine")

st.sidebar.header("Téléchargez vos données de test")

uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, skipinitialspace=True)
    
    # Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip()  # Supprimer les espaces inutiles
    
    if 'IndividualID' in df.columns:
        df.rename(columns={'IndividualID': 'ID'}, inplace=True)
    
    # Vérifier la présence de toutes les colonnes nécessaires
    required_columns = ['Min:Sec'] + [f'D{i}' for i in range(1, 65)] + ['ID']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Le fichier téléchargé ne contient pas les colonnes obligatoires : {', '.join(missing_columns)}")
    else:
        records = df[required_columns].to_dict(orient='records')
        
        if st.sidebar.button("Prédire"):
            with st.spinner("Analyse des données en cours..."):
                headers = {'Content-Type': 'application/json'}
                response = requests.post("https://e-nose-nasa-covid-19-pr-diction-9.onrender.com/predict", data=json.dumps(records), headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    predictions = result.get("predictions", [])
                    if predictions:
                        df["Prédiction"] = predictions
                        st.write("Résultats de la prédiction :")
                        st.write(df)
                    else:
                        st.error("Aucune prédiction reçue du backend.")
                else:
                    st.error(f"Erreur du backend : {response.status_code}, {response.text}")
