import streamlit as st
import pandas as pd
import requests
import io

st.set_page_config(
    page_title="Breath Diagnostics Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Breath Diagnostics Prediction :chart_with_upwards_trend:")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #fff;
        border-right: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Upload Test Data")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    st.write("Preview of the uploaded file:")
    st.write(df.head())

    # Extract relevant columns
    columns = ['Min:Sec'] + [f'D{i}' for i in range(1, 65)]
    if not all(col in df.columns for col in columns):
        st.error("Uploaded file is missing required columns.")
    else:
        # Prepare data for API request
        records = df[columns].to_dict(orient='records')

        if st.sidebar.button("Predict"):
            with st.spinner("Analyzing the input..."):
                # Send the data to the backend API
                response = requests.post("https://e-nose-nasa-covid-19-pr-diction-9.onrender.com", json=records)
                result = response.json()

                # Display the results
                predictions = result.get("predictions", [])
                if predictions:
                    df["Prediction"] = predictions
                    st.write("Prediction Results:")
                    st.write(df)
                else:
                    st.error("No predictions received from the backend.")
