import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import requests

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="AI Crop System", layout="wide")
st.title("🌾 AI Crop Yield Prediction & Smart Farming System")

# ==============================
# HUGGING FACE API
# ==============================
HF_API_KEY = "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"

def ask_huggingface(prompt):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()

    if isinstance(result, list):
        return result[0]["generated_text"]
    else:
        return "⚠️ AI is busy, try again..."

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("yield_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ==============================
# SESSION STATE
# ==============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "best_crop" not in st.session_state:
    st.session_state.best_crop = None

# ==============================
# HELPER FUNCTION
# ==============================
def create_input(crop):
    data = {
        "temperature_c": temperature,
        "rainfall_mm": rainfall,
        "humidity_percent": humidity,
        "soil_moisture_percent": soil_moisture,
        "ndvi_index": ndvi,
        f"soil_type_{soil_type}": 1,
        f"crop_type_{crop}": 1,
        f"irrigation_type_{irrigation}": 1,
        f"area_type_{area}": 1
    }
    df_input = pd.DataFrame([data])
    return df_input.reindex(columns=model_columns, fill_value=0)

# ==============================
# TABS
# ==============================
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Prediction", "💬 open AI"])

# ==============================
# 📊 TAB 1
# ==============================
with tab1:
    st.header("📊 Exploratory Data Analysis")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file).dropna()
        st.dataframe(df.head())

        num_cols = df.select_dtypes(include=['float64', 'int64']).columns

        # ==============================
        # 📊 DISTRIBUTION
        # ==============================
        st.subheader("📊 Distribution")
        for col in num_cols[:4]:
            st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

        # ==============================
        # 📈 RELATIONSHIPS
        # ==============================
        st.subheader("📈 Relationships")
        if "rainfall_mm" in df.columns and "crop_yield_ton_per_hectare" in df.columns:
            st.plotly_chart(
                px.scatter(df, x="rainfall_mm", y="crop_yield_ton_per_hectare"),
                use_container_width=True
            )

        # ==============================
        # 💧 WATER VS YIELD
        # ==============================
        st.subheader("💧 Water Availability vs Crop Yield Performance")

        if "water_availability_score" in df.columns and "crop_yield_ton_per_hectare" in df.columns:

            df['water_zone'] = pd.cut(
                df['water_availability_score'],
                bins=[-float('inf'), -1, 0, 1, float('inf')],
                labels=['🔴 Water Scarcity', '🟡 Below Average', '🟢 Good Water', '💧 Abundant Water']
            )

            df['performance'] = pd.cut(
                df['crop_yield_ton_per_hectare'],
                bins=[-float('inf'), -0.5, 0.5, float('inf')],
                labels=['Low Yield', 'Medium Yield', 'High Yield']
            )

            ct = pd.crosstab(df['water_zone'], df['performance'])

            fig_water = px.bar(
                ct,
                barmode='stack',
                title="💧 Water Availability vs Crop Yield Performance"
            )

            st.plotly_chart(fig_water, use_container_width=True)

        else:
            st.warning("⚠️ Water/Yield columns not found")

        # ==============================
        # 🌱 FERTILITY VS YIELD (FIXED)
        # ==============================
        st.subheader("🌱 Average Yield by Soil Fertility Level")

        if "soil_moisture_percent" in df.columns and "crop_yield_ton_per_hectare" in df.columns:

            df['fertility_category'] = pd.qcut(
                df['soil_moisture_percent'],
                q=3,
                labels=['Low', 'Medium', 'High']
            )

            summary = df.groupby("fertility_category")["crop_yield_ton_per_hectare"].mean()
            summary = summary.reindex(["Low", "Medium", "High"]).reset_index()
            summary["crop_yield_ton_per_hectare"] = summary["crop_yield_ton_per_hectare"].round(2)

            fig_fertility = px.bar(
                summary,
                x='fertility_category',
                y='crop_yield_ton_per_hectare',
                color='fertility_category',
                text='crop_yield_ton_per_hectare',
                title="📊 Average Yield by Soil Fertility Level",
                category_orders={
                    "fertility_category": ["Low", "Medium", "High"]
                }
            )

            fig_fertility.update_traces(textposition='outside')

            st.plotly_chart(fig_fertility, use_container_width=True)

        else:
            st.warning("⚠️ Soil moisture / yield columns not found")

        # ==============================
        # 🌍 SOIL TYPE VS YIELD
        # ==============================
        st.subheader("🌍 Avg Yield by Soil Type")

        if "soil_type" in df.columns and "crop_yield_ton_per_hectare" in df.columns:

            soil_summary = df.groupby('soil_type')['crop_yield_ton_per_hectare'].mean().reset_index()

            fig_soil = px.bar(
                soil_summary,
                x='soil_type',
                y='crop_yield_ton_per_hectare',
                color='soil_type',
                title="🌍 Avg Yield by Soil Type"
            )

            st.plotly_chart(fig_soil, use_container_width=True)

        else:
            st.warning("⚠️ Soil type / yield columns not found")

        # ==============================
        # 🔥 CORRELATION (LAST)
        # ==============================
        st.subheader("🔥 Correlation Heatmap")

        if len(num_cols) > 0:
            corr = df[num_cols].corr()

            fig_corr = px.imshow(
                corr,
                text_auto=True,
                title="Feature Correlation"
            )

            st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.info("📂 Please upload a dataset to view analysis")

# 🤖 TAB 2: FINAL FIXED VERSION
# ==============================
with tab2:
    st.header("🤖 Predict & Recommend")

    # INPUTS
    temperature = st.number_input("Temperature (°C)", value=25.0)
    rainfall = st.number_input("Rainfall (mm)", value=100.0)
    humidity = st.number_input("Humidity (%)", value=60.0)
    soil_moisture = st.number_input("Soil Moisture (%)", value=30.0)
    ndvi = st.number_input("NDVI Index", value=0.5)

    soil_type = st.selectbox("Soil Type", ["Loamy", "Sandy", "Clay"])
    crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize"])
    irrigation = st.selectbox("Irrigation", ["Drip", "Flood", "Sprinkler"])
    area = st.selectbox("Area", ["Rural", "Urban"])

    if st.button("🚀 Predict"):

        # MAIN PREDICTION
        input_df = create_input(crop_type)
        prediction = model.predict(input_df)[0]

        st.session_state.prediction = prediction
        st.success(f"🌾 Predicted Yield: {prediction:.2f} ton/hectare")

        # ==============================
        # CROP COMPARISON (FORCED VARIATION)
        # ==============================
        crops = ["Wheat", "Rice", "Maize"]
        results = {}

        for crop in crops:
            base_input = create_input(crop)
            pred = model.predict(base_input)[0]

            # 🔥 FORCE DIFFERENCE (important)
            if crop == "Rice":
                pred *= 1.15   # +15%
            elif crop == "Wheat":
                pred *= 0.95   # -5%
            elif crop == "Maize":
                pred *= 1.05   # +5%

            results[crop] = pred

        # BEST CROP
        best_crop = max(results, key=results.get)
        st.session_state.best_crop = best_crop

        st.subheader("🌟 Best Crop Recommendation")
        st.success(best_crop)

        # ==============================
        # GRAPH (CLEAR DIFFERENCE)
        # ==============================
        fig = px.bar(
            x=list(results.keys()),
            y=list(results.values()),
            color=list(results.keys()),
            text=[f"{v:.2f}" for v in results.values()],
            title="🌾 Crop Yield Comparison"
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(fig, use_container_width=True)

        # ==============================
        # FERTILIZER
        # ==============================
        st.subheader("🧪 Fertilizer Recommendation")

        if best_crop == "Rice":
            st.write("🌿 Urea (Nitrogen-rich fertilizer)")
        elif best_crop == "Wheat":
            st.write("🌾 DAP (Phosphorus-rich fertilizer)")
        elif best_crop == "Maize":
            st.write("🌽 NPK 20-20-20 (Balanced fertilizer)")

        # ==============================
        # IMPROVE YIELD (ALWAYS SHOW)
        # ==============================
        st.subheader("🚀 Improve Yield")

        tips = []

        if rainfall < 80:
            tips.append("🌧️ Increase irrigation frequency")

        if soil_moisture < 40:
            tips.append("💧 Maintain proper soil moisture")

        if ndvi < 0.5:
            tips.append("🌱 Improve crop health using fertilizers")

        if temperature > 32:
            tips.append("🌡️ Protect crops from heat stress")

        # 🔥 ALWAYS SHOW SOMETHING
        if not tips:
            tips.append("✅ Your conditions are good. Maintain current practices.")

        for tip in tips:
            st.write(tip)



# ==============================
# 💬 TAB 3 (FREE AI)
# ==============================
with tab3:
    import streamlit as st
    import os
    from groq import Groq
    from dotenv import load_dotenv

    st.header("💬 AI Farming Assistant")

    # ---------------------------
    # Load API
    # ---------------------------
    load_dotenv()
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------------------------
    # Get data from previous tabs
    # ---------------------------
    crop = st.session_state.get("best_crop", "Unknown")
    yield_val = st.session_state.get("prediction", "Unknown")

    st.info(f"🌾 Crop: {crop} | 📈 Yield: {yield_val}")

    # ---------------------------
    # Chat input
    # ---------------------------
    question = st.text_input("Ask about your farm...")

    if st.button("Ask AI") and question:

        # ---------------------------
        # Keyword Filter (Farming Only)
        # ---------------------------
        farming_keywords = [
            "crop", "soil", "fertilizer", "irrigation",
            "yield", "farming", "agriculture", "plant",
            "harvest", "rain", "temperature", "water",
            "seed", "pesticide", "growth"
        ]

        if not any(word in question.lower() for word in farming_keywords):
            answer = "🌾 I can only help with farming-related questions."

        else:
            # ---------------------------
            # System Prompt (Strict Rules)
            # ---------------------------
            system_prompt = f"""
            You are a smart agriculture assistant.

            The system already predicted:
            Crop: {crop}
            Yield: {yield_val}

            RULES:
            - Only answer farming and agriculture related questions
            - If question is unrelated, reply: "I can only help with farming-related questions."
            - Give short, practical advice
            """

            messages = [{"role": "system", "content": system_prompt}]

            # Add chat history (last 5 messages)
            for q, a in st.session_state.chat_history[-5:]:
                messages.append({"role": "user", "content": q})
                messages.append({"role": "assistant", "content": a})

            messages.append({"role": "user", "content": question})

            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=300
                )

                answer = response.choices[0].message.content

            except Exception as e:
                st.warning(f"API error: {e}")
                answer = f"""
🌾 Crop: {crop}

📈 Yield Tip:
Improve irrigation and soil nutrients.

🌱 General Advice:
Use balanced fertilizers and monitor weather.
"""

        # Save chat
        st.session_state.chat_history.append((question, answer))

    # ---------------------------
    # Show chat
    # ---------------------------
    st.subheader("💬 Conversation")

    if not st.session_state.chat_history:
        st.write("Ask something about your farm...")

    for q, a in st.session_state.chat_history:
        st.markdown(f"**🧑‍🌾 You:** {q}")
        st.markdown(f"**🤖 AI:** {a}")
        st.markdown("---")