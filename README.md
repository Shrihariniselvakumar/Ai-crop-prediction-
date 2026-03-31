# 🌾 Smart Farming Assistant using AI & Machine Learning

## 📌 Overview

The **Smart Farming Assistant** is an AI-powered web application built using **Machine Learning and Generative AI** to help farmers make better decisions.

It predicts:

* 🌾 Best crop to cultivate
* 📈 Expected crop yield

And provides:

* 🤖 AI-based farming advice through an interactive chatbot

---

## 🎯 Features

### 🌱 Crop Recommendation

* Suggests the most suitable crop based on soil and environmental conditions.

### 📊 Yield Prediction

* Predicts crop yield using machine learning models.

### 💬 AI Farming Assistant (Chatbot)

* Answers user queries based on:

  * Predicted crop
  * Yield results
  * Farming conditions
* Provides real-time, practical agricultural advice.

### 📈 Interactive Dashboard

* User-friendly interface built with Streamlit
* Displays predictions and insights clearly

---

## 🧠 Technologies Used

### 👨‍💻 Programming

* Python

### 📊 Machine Learning

* Scikit-learn
* XGBoost

### 🤖 AI / LLM

* LLaMA 3 via Groq

### 🌐 Web Framework

* Streamlit

### 🗄 Dataset Handling

* Pandas, NumPy

---

## ⚙️ How It Works

### 1️⃣ Data Processing

* Agricultural dataset is cleaned and preprocessed
* Features include:

  * Temperature
  * Rainfall
  * Humidity
  * Soil moisture
  * NDVI

---

### 2️⃣ Model Training

* Machine Learning models are trained using historical data
* **XGBoost Regressor** is used for accurate yield prediction

---

### 3️⃣ Prediction System

* User inputs environmental data
* System predicts:

  * Best crop 🌾
  * Expected yield 📈

---

### 4️⃣ AI Chat Assistant

* Uses LLM (LLaMA 3) via Groq API
* Combines:

  * ML predictions
  * User queries
* Provides intelligent farming advice

---

## 🖥 Application Structure

### 🔹 Tab 1 & 2

* Crop recommendation
* Yield prediction

### 🔹 Tab 3

* AI chatbot
* Uses prediction results to answer user questions

---

## 🚀 Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone <your-repo-link>
cd project-folder
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate      # windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 💡 Example Usage

1. Enter farm conditions
2. Get:

   * 🌾 Recommended crop
   * 📈 Predicted yield
3. Ask AI:

   * “How to improve yield?”
   * “Which fertilizer should I use?”

---

## 🔮 Future Enhancements

* 🌦 Real-time weather integration
* 📊 Advanced analytics & visualization
* 🌍 Multi-language support
* 📱 Mobile application

---

## 🎓 Conclusion

This project demonstrates how **Machine Learning + AI** can transform agriculture by providing **data-driven insights and intelligent assistance** to farmers.

---

## 👩‍💻 Author

Developed by **ShrihariniSelvakumar** 🌟
