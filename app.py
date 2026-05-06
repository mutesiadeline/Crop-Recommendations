import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('lightgbm.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make prediction
def predict_crop_type(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit App
def main():
    st.title("Crop Type Prediction")

    st.write("This app predicts the crop type based on various soil and environmental parameters.")

    # Input fields
    N = st.number_input("Nitrogen content (kg/ha)", min_value=0.0, value=20.0, step=1.0)
    P = st.number_input("Phosphorous content (kg/ha)", min_value=0.0, value=10.0, step=1.0)
    K = st.number_input("Potassium content (kg/ha)", min_value=0.0, value=30.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    ph = st.number_input("Soil pH value", min_value=0.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0, step=1.0)

    if st.button("Predict"):
        prediction = predict_crop_type(N, P, K, temperature, humidity, ph, rainfall)
        st.write(f"Predicted Crop Type: {prediction}")

# Run the app
if __name__ == "__main__":
    main()
