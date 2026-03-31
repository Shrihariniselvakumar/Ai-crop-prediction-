import streamlit as st
import pandas as pd
import joblib
import os
from openai import OpenAI
from dotenv import load_dotenv

# ==============================
# Load API Key
# ==============================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==============================
# Load Model
# ==============================
model = joblib.load("yield_model.pkl")
columns = joblib.load("model_columns.pkl")

# ==============================
# Streamlit UI
# ==============================
st.title("🌾 Crop Yield Prediction + AI Advisor")

# User Inputs
rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0)

soil = st.selectbox("Soil Type", ["Loamy", "Sandy", "Clay"])
crop = st.selectbox("Crop Type", ["Rice", "Wheat", "Maize"])
irrigation = st.selectbox("Irrigation", ["Drip", "Flood", "Sprinkler"])
area = st.selectbox("Area Type", ["Rural", "Urban"])

# ==============================
# Prediction Button
# ==============================
if st.button("Predict Yield"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "rainfall": [rainfall],
        "temperature": [temperature],
        "humidity": [humidity],
        "soil_type": [soil],
        "crop_type": [crop],
        "irrigation_type": [irrigation],
        "area_type": [area]
    })

    # Encode
    input_data = pd.get_dummies(input_data)

    # Align columns
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(f"🌾 Predicted Yield: {prediction:.2f} tons/hectare")

    # ==============================
    # OpenAI Explanation
    # ==============================
    with st.spinner("🤖 Generating AI advice..."):

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert agriculture advisor."
                },
                {
                    "role": "user",
                    "content": f"""
                    The predicted crop yield is {prediction:.2f} tons per hectare.

                    Conditions:
                    - Rainfall: {rainfall}
                    - Temperature: {temperature}
                    - Humidity: {humidity}
                    - Soil: {soil}
                    - Crop: {crop}
                    - Irrigation: {irrigation}

                    Give suggestions to improve yield.
                    """
                }
            ]
        )

        advice = response.choices[0].message.content

    st.subheader("🤖 AI Recommendations")
    st.write(advice)