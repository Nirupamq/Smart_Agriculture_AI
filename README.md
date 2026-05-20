# Smart Agriculture AI Advisor 🌱🤖

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-green)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![Status](https://img.shields.io/badge/Project-Completed-success)

## Overview

Smart Agriculture AI Advisor is an AI-powered crop recommendation system designed to help farmers make informed agricultural decisions. The application analyzes environmental and soil conditions to predict the most suitable crop and provide profitable crop recommendations for a given region.

The system leverages Machine Learning techniques to transform traditional farming decisions into data-driven recommendations, improving productivity and profitability.

---

## Problem Statement

Farmers often face challenges in selecting the right crop due to varying soil conditions, rainfall patterns, temperature changes, and market factors.

This project aims to solve that problem by:

- Predicting the most suitable crop for a region
- Analyzing soil and environmental conditions
- Providing AI-based recommendations
- Suggesting profitable crop options

---

## Features

### 🌾 Crop Prediction

Predicts the best crop based on:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH Level
- Rainfall

### 💰 Profitability Analysis

Provides recommendations on crops that may offer better economic returns.

### 📊 Explainable AI

Displays important factors influencing crop recommendations.

### 📈 Data Visualization

Visualizes agricultural data and model insights.

### 🖥️ User-Friendly Interface

Simple interface for entering agricultural parameters and viewing recommendations.

---

## Technology Stack

### Programming Language

- Python

### Libraries Used

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

### Machine Learning Algorithms

- Random Forest Classifier
- Decision Tree Classifier

### Model Evaluation Metrics

- Accuracy Score
- Precision
- Recall
- F1 Score

---

## System Architecture

```text
Input Data
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Machine Learning Model
    ↓
Crop Prediction
    ↓
Profit Recommendation
    ↓
User Dashboard
```

---

## Dataset Features

| Feature | Description |
|----------|-------------|
| N | Nitrogen Content |
| P | Phosphorus Content |
| K | Potassium Content |
| Temperature | Average Temperature (°C) |
| Humidity | Relative Humidity (%) |
| pH | Soil pH Value |
| Rainfall | Rainfall (mm) |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/smart-agriculture-ai.git
cd smart-agriculture-ai
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Project Workflow

1. Collect agricultural data
2. Preprocess and clean data
3. Train machine learning model
4. Predict suitable crops
5. Analyze profitability
6. Generate recommendations
7. Display results through dashboard

---

## Future Enhancements

- Real-time weather API integration
- Satellite data analysis
- Disease prediction module
- Fertilizer recommendation system
- Mobile application support
- Multi-language support for farmers

---

## Results

The system successfully predicts suitable crops using agricultural parameters and provides intelligent recommendations that can help improve farming decisions and profitability.

---

## Applications

- Precision Agriculture
- Smart Farming
- Agricultural Decision Support Systems
- Government Agricultural Programs
- Farm Management Solutions

---

## Author

SADHANALA NIRUPAMA CHOWDARY

AI-Based Smart Agriculture Project

---

## License

This project is developed for educational and research purposes.