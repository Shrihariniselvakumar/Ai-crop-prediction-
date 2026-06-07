# 🌾 AI Crop Prediction using Climate and Soil Data Analytics

> An end-to-end intelligent crop recommendation system powered by **XGBoost** and a **LLaMA 3 chatbot** — giving smallholder farmers real-time, soil-specific farming guidance.

---

## 📌 Overview

This project combines machine learning and generative AI to help farmers make data-driven crop decisions. Given climate and soil parameters, the system recommends the most suitable crop and provides instant advisory through a conversational AI chatbot.

---

## 📸 App Screenshots
<img width="607" height="290" alt="Screenshot 2026-06-07 184911" src="https://github.com/user-attachments/assets/dfc7f953-8a39-48f9-a998-cd3b09dd8599" />


### 🔍 Exploratory Data Analysis
![EDA Dashboard](screenshots/eda_dashboard.png)
> Upload your dataset and instantly explore patterns — the EDA tab processes CSV files and surfaces key agricultural insights.

### 📊 Water vs Yield Analysis
<img width="603" height="335" alt="Screenshot 2026-06-07 185022" src="https://github.com/user-attachments/assets/93f0f3f2-43e2-454d-803c-b6c75b7ae86d" />

![Water vs Yield Chart](screenshots/water_vs_yield.png)
> Stacked bar chart showing how water availability (Scarcity → Abundant) correlates with crop yield levels (Low / Medium / High).

### 🤖 Crop Prediction Input Panel
<img width="604" height="340" alt="Screenshot 2026-06-07 190325" src="https://github.com/user-attachments/assets/1a202d68-dbdd-4208-ad62-bfa008900338" />

![Prediction Panel](screenshots/prediction_panel.png)
> Input your soil type, crop type, irrigation method, and area — the model predicts yield (ton/hectare) and recommends the best crop instantly.

### 📈 Feature Importance Chart
<img width="608" height="345" alt="Screenshot 2026-06-07 190342" src="https://github.com/user-attachments/assets/1fb5b999-5bf4-4046-bf95-dd462fcc5b84" />

![Feature Importance](screenshots/feature_importance.png)
> XGBoost feature importance plot revealing which soil and climate factors drive yield predictions the most.

---

## ✨ Features

- 🌱 **Crop Recommendation** — XGBoost model trained on climate + soil datasets, predicting across 5+ crop types with high accuracy
- 🤖 **Smart Farming Chatbot** — Groq-powered LLaMA 3 chatbot for real-time, context-aware farming advice
- 📊 **Interactive EDA Dashboard** — Upload any CSV dataset and explore correlations, distributions, and trends
- 📉 **Feature Importance Visualization** — Understand which factors influence crop yield the most
- 🌍 **Targeted for Smallholder Farmers** — Simple, actionable guidance without needing technical knowledge

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost, Scikit-learn |
| Data Processing | Python, Pandas, NumPy |
| AI Chatbot | Groq API (LLaMA 3) |
| Frontend / UI | Streamlit |
| LLM Integration | OpenAI API |

---

## 📁 Project Structure

```
Ai-crop-prediction/
│
├── app.py                  # Streamlit app entry point
├── model/
│   ├── train_model.py      # XGBoost training script
│   └── crop_model.pkl      # Saved trained model
├── data/
│   └── crop_dataset.csv    # Climate and soil dataset
├── chatbot/
│   └── groq_chat.py        # LLaMA 3 chatbot via Groq API
├── screenshots/
│   ├── eda_dashboard.png
│   ├── water_vs_yield.png
│   ├── prediction_panel.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Shrihariniselvakumar/Ai-crop-prediction-.git
cd Ai-crop-prediction-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🧠 How It Works

```
User inputs soil & climate data
        ↓
XGBoost model predicts yield & best crop
        ↓
LLaMA 3 chatbot provides farming advice
        ↓
Streamlit displays results & visualizations
```

1. **EDA Tab** — Upload your dataset; the app auto-generates Water vs Yield charts, correlation heatmaps, and distribution plots
2. **Prediction Tab** — Enter soil type, crop type, irrigation method, and area; XGBoost predicts yield in ton/hectare
3. **AI Insights Tab** — LLaMA 3 via Groq API generates context-aware advisory — fertilizer tips, watering schedules, seasonal guidance

---

## 🌐 Input Parameters

| Parameter | Options |
|-----------|---------|
| State | Region dropdown |
| NDV | Vegetation index |
| Soil Type | Loamy, Sandy, Clay, etc. |
| Crop Type | Wheat, Rice, Maize, etc. |
| Irrigation | Drip, Flood, Sprinkler, etc. |
| Area | Rural / Semi-urban / Urban |

---

## 📊 Model Performance

| Metric | Detail |
|--------|--------|
| Model | XGBoost Classifier + Regressor |
| Crops Supported | 5+ crop types |
| Output | Predicted yield (ton/hectare) + Best Crop |
| Visualization | Feature importance, Water vs Yield, Correlation |

---

## 🔮 Future Improvements

- [ ] Real-time weather API integration (OpenWeatherMap)
- [ ] Regional language responses in chatbot (Tamil, Hindi)
- [ ] Deploy on Streamlit Cloud / HuggingFace Spaces
- [ ] Fertilizer quantity recommendation module
- [ ] Mobile-friendly PWA version

---

## 👩‍💻 Author

**Shri Harini Selvakumar**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shriharini-selvakumar-8811aa361)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Shrihariniselvakumar)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
