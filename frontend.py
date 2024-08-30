import streamlit as st
import requests
import pandas as pd
from io import StringIO

st.set_page_config(
    page_title="Prédiction du diagnostic de l’haleine 📈",
    page_icon=":chart_with_upwards_trend:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Prédiction du diagnostic de l’haleine 📈")

st.sidebar.header("Téléchargez vos données de test")

uploaded_file = st.sidebar.file_uploader("Choisissez un fichier TXT", type="txt")

def extract_id_and_load_data(file_content):
    # Convertir le contenu du fichier en DataFrame
    df = pd.read_csv(StringIO(file_content), delimiter='\t', skiprows=1)

    # Lire la première ligne pour obtenir l'identifiant unique
    lines = file_content.split('\n')
    first_line = lines[0].strip()

    # Vérifier si la première ligne contient un identifiant valide
    if first_line.startswith('ID:'):
        # Extraire l'identifiant après 'ID:'
        individual_id = first_line.split(':')[1].strip()
    else:
        # Si le format n'est pas correct, utiliser un identifiant par défaut
        individual_id = "Unknown_ID"
        st.warning(f"Avertissement : Aucun ID valide trouvé. Utilisation de 'Unknown_ID'.")

    # Ajouter l'identifiant au DataFrame
    df['IndividualID'] = individual_id

    # Renommer les colonnes
    df.rename(columns={'IndividualID': 'ID'}, inplace=True)

    return df

if uploaded_file:
    st.write("Fichier TXT téléchargé avec succès.")

    # Lire le contenu du fichier TXT
    txt_content = uploaded_file.read().decode("utf-8")

    # Extraire les données du fichier
    try:
        test_df = extract_id_and_load_data(txt_content)

        # Vérifiez les noms de colonnes disponibles
        st.write("Colonnes disponibles dans le fichier :")
        st.write(test_df.columns.tolist())

        # Vérifier la présence de la colonne 'Min:Sec'
        if 'Min:Sec' not in test_df.columns:
            st.error("La colonne 'Min:Sec' est manquante dans le fichier.")
        else:
            # Afficher les premières lignes du DataFrame pour vérification
            st.write("Aperçu des données du fichier téléchargé :")
            st.write(test_df.head())

            # Convertir le DataFrame en CSV (en mémoire)
            csv_buffer = StringIO()
            test_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)  # Revenir au début du buffer

            # Envoyer le fichier CSV au backend pour prédiction
            if st.sidebar.button("Prédire"):
                try:
                    response = requests.post(
                        "http://localhost:8000/predict",  # Corriger l'URL pour appeler le bon endpoint
                        files={"file": csv_buffer}
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
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {str(e)}")
