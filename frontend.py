import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="Breath Diagnostics Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Breath Diagnostics Prediction :chart_with_upwards_trend:")

st.sidebar.header("Upload Test Data")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Vérifier et renommer la colonne IndividualID en ID
    if 'IndividualID' in df.columns:
        df.rename(columns={'IndividualID': 'ID'}, inplace=True)

    st.write("Preview of the uploaded file:")
    st.write(df.head())

    # Extract relevant columns
    columns = ['ID', 'Min:Sec'] + [f'D{i}' for i in range(1, 65)]
    if not all(col in df.columns for col in columns):
        st.error("Uploaded file is missing required columns.")
    else:
        # Prepare data for API request
        records = df[columns].to_dict(orient='records')

        if st.sidebar.button("Predict"):
            with st.spinner("Analyzing the input..."):
                # Send the data to the backend API
                response = requests.post("https://e-nose-nasa-covid-19-pr-diction-9.onrender.com/predict", json=records)
                if response.status_code == 200:
                    result = response.json()

                    # Display the results
                    predictions = result.get("predictions", [])
                    if predictions:
                        df["Prediction"] = predictions
                        st.write("Prediction Results:")
                        st.write(df)
                    else:
                        st.error("No predictions received from the backend.")
                else:
                    st.error(f"Error from backend: {response.status_code}")
