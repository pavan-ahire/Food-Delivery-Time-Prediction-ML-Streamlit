# 🚚 Food Delivery Time Prediction using Machine Learning & Streamlit

This project predicts **food delivery time** based on order and delivery-related factors using **Machine Learning regression models** and a **Streamlit web application** for real-time predictions.

---

## 🔗 Project Links

- 🌐 **Live Streamlit App**:  
  https://food-delivery-time-prediction-ml-app-qkesjwapjuowuwfqqybvpw.streamlit.app/

- 📁 **GitHub Repository**:  
  https://github.com/pavan-ahire/Food-Delivery-Time-Prediction-ML-Streamlit

---

## 📌 Project Overview

The objective of this project is to estimate the **delivery time (in minutes)** for food orders using historical delivery data and machine learning techniques.

The model takes into account factors such as distance, traffic, weather, preparation time, and delivery conditions to generate predictions.

---
## 📂 Project Structure

```text
Food-Delivery-Time-Prediction-ML-Streamlit/
│
├── app.py
│   └── Streamlit web application for delivery time prediction
│
├── Delivery time prediction ML_streamlit.ipynb
│   └── Jupyter notebook containing EDA, preprocessing, model training, and evaluation
│
├── delivery_time_model.pkl
│   └── Trained machine learning regression model
│
├── scaler.pkl
│   └── StandardScaler used for feature scaling during training and prediction
│
├── food_delivery_times.csv
│   └── Dataset used for training and testing the model
│
├── requirements.txt
│   └── Python dependencies required to run the project
│
└── README.md
    └── Project documentation

```
---

## 📊 Dataset

- The dataset contains historical food delivery information.
- The target variable is **delivery time in minutes**.
- Data preprocessing and cleaning were performed before training.

---

## 🔍 Exploratory Data Analysis (EDA)

The following analysis was performed:

- Univariate analysis of numerical features
- Bivariate analysis between delivery time and key features
- Understanding feature distributions

---

## 🧠 Machine Learning Workflow

- Feature selection and preprocessing
- Manual encoding of categorical variables
- Train-test split
- Feature scaling using **StandardScaler**
- Regression model training and evaluation
- Saving the trained model and scaler using `joblib`

> The same scaler used during training is reused during prediction to ensure accurate results.

---

## 🌐 Streamlit Application

The Streamlit app allows users to:

- Input delivery-related details
- Predict estimated delivery time
- View predictions in minutes or hours format

Key points:
- Model and scaler loaded using caching
- Correct scaling applied during inference
- User-friendly interface

---
