import streamlit as st
import pandas as pd
import joblib

model = joblib.load("car_price_model.pkl")
feature_names = joblib.load("feature.pkl")

st.title("Car Price Prediction")
st.write("Predict the price of a Ford car")

year = st.number_input("Year", min_value=1996, max_value=2025, value=2018)

mileage = st.number_input("Mileage", min_value=0, value=10000)

tax = st.number_input("Tax", min_value=0, value=150)

mpg = st.number_input("MPG", min_value=0.0, value=55.0)

engineSize = st.number_input("Engine Size", min_value=0.0, value=1.0)

model_name = st.selectbox(
    "Model",
    ["Fiesta","Focus","Kuga","EcoSport","Mondeo"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual","Automatic","Semi-Auto"]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol","Diesel","Hybrid","Electric","Other"]
)

predict_btn = st.button("Predict Price")
if predict_btn:

    input_data = {
        'year': year,
        'mileage': mileage,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engineSize
    }

    input_df = pd.DataFrame([input_data])

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    model_col = f"model_{model_name}"
    transmission_col = f"transmission_{transmission}"
    fuel_col = f"fuelType_{fuel_type}"

    if model_col in input_df.columns:
        input_df[model_col] = 1

    if transmission_col in input_df.columns:
        input_df[transmission_col] = 1

    if fuel_col in input_df.columns:
        input_df[fuel_col] = 1

    input_df = input_df[feature_names]
    st.write(input_df)

    prediction = model.predict(input_df)

    st.success(f"Predicted Price = £{prediction[0]:,.2f}")