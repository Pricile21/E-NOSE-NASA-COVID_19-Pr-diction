import streamlit as st
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
    st.write("Fichier téléchargé avec succès.")
    
    # Envoyer le fichier au backend pour prédiction
    if st.sidebar.button("Prédire"):
        try:
            # Convertir le fichier téléchargé en un fichier compatible pour la requête
            response = requests.post(
                "https://e-nose-nasa-covid-19-pr-diction-3.onrender.com/predict",
                files={"file": uploaded_file.getvalue()}
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])
                if predictions:
                    st.write("Résultats de la prédiction :")
                    st.write(predictions)
                else:
                    st.error("Aucune prédiction reçue du backend.")
            else:
                st.error(f"Erreur du backend : {response.status_code}, {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de l'envoi de la requête : {str(e)}")
