import streamlit as st
import pandas as pd
import requests

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
    # Lire le fichier téléchargé en corrigeant le problème de virgule potentielle
    df = pd.read_csv(uploaded_file, skipinitialspace=True)
    
    # Vérifiez si la première colonne a été décalée par une virgule
    if df.columns[0].startswith(','):
        # Si une virgule est trouvée, la supprimer
        df.columns = df.columns.str.lstrip(',')
    
    # Renommer la colonne 'IndividualID' en 'ID' si nécessaire
    if 'IndividualID' in df.columns:
        df.rename(columns={'IndividualID': 'ID'}, inplace=True)
    
    # Transformer 'Min:Sec' en secondes
    df['Min:Sec'] = df['Min:Sec'].apply(lambda x: int(x.split(':')[0]) * 60 + float(x.split(':')[1]))

    # Regrouper par 'ID' et calculer la moyenne
    df = df.groupby('ID').mean().reset_index()

    st.write("Aperçu du fichier téléchargé :")
    st.write(df.head())

    # Définir les colonnes attendues
    required_columns = ['Min:Sec'] + [f'D{i}' for i in range(1, 65)] + ['ID']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Le fichier téléchargé ne contient pas les colonnes obligatoires : {', '.join(missing_columns)}")
    else:
        # Préparer les données pour la requête API
        records = df[required_columns].to_dict(orient='records')

        if st.sidebar.button("Prédire"):
            with st.spinner("Analyse des données en cours..."):
                # Envoyer les données à l'API backend
                response = requests.post("https://e-nose-nasa-covid-19-pr-diction-9.onrender.com/predict", json=records)
                if response.status_code == 200:
                    result = response.json()

                    # Afficher les résultats
                    predictions = result.get("predictions", [])
                    if predictions:
                        df["Prédiction"] = predictions
                        st.write("Résultats de la prédiction :")
                        st.write(df)
                    else:
                        st.error("Aucune prédiction reçue du backend.")
                else:
                    st.error(f"Erreur du backend : {response.status_code}")
