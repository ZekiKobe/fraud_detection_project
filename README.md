# 🚨 Fraud Detection System - Adey Innovations Inc.

## 📌 Project Overview

This end-to-end fraud detection solution addresses critical business
needs:

-   **E-commerce transaction security**
-   **Credit card fraud prevention**
-   **Real-time risk assessment**

### ✅ Key Features

-   Geolocation analysis via IP mapping
-   Behavioral pattern recognition
-   Explainable AI using SHAP interpretations
-   Optimized handling of class imbalance

------------------------------------------------------------------------

## 🏗 Technical Architecture

``` mermaid
graph LR
    A[Raw Data] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[API Deployment]
    E --> F[Monitoring Dashboard]
```

------------------------------------------------------------------------

## 📂 Repository Structure

    fraud-detection/
    ├── data/
    │   ├── raw/               # Original datasets
    │   └── processed/         # Cleaned data
    ├── notebooks/
    │   ├── 01_EDA.ipynb       # Exploratory analysis
    │   ├── 02_Modeling.ipynb  # Model development
    │   └── 03_SHAP.ipynb      # Explainability
    ├── src/
    │   ├── preprocessing.py   # Data cleaning
    │   ├── features.py        # Feature engineering
    │   └── models.py          # ML pipelines
    ├── models/                # Saved models
    └── reports/               # Output visualizations

------------------------------------------------------------------------

## 🔎 Task 1: Data Analysis & Preprocessing

### Data Cleaning

``` python
df = df.dropna(subset=['purchase_value', 'ip_address'])
df['signup_time'] = pd.to_datetime(df['signup_time'])
```

### Feature Engineering

``` python
df['country'] = df['ip_address'].apply(map_ip_to_country)
df['time_since_signup'] = (
    df['purchase_time'] - df['signup_time']
).dt.total_seconds() / 3600
```

### Class Imbalance Handling

``` python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.1, random_state=42)
```

------------------------------------------------------------------------

## 🤖 Task 2: Model Development

### 📊 Model Comparison

  Model                 Precision   Recall   F1-Score   AUC-PR
  --------------------- ----------- -------- ---------- --------
  Logistic Regression   0.78        0.65     0.71       0.82
  XGBoost               0.89        0.73     0.80       0.91

**Final Selection:** XGBoost demonstrated superior precision and AUC-PR
performance --- critical metrics for fraud detection systems.

------------------------------------------------------------------------

## 🔍 Task 3: Model Explainability

### SHAP Analysis Insights

-   Transactions within 24 hours of signup significantly increase fraud
    risk
-   High purchase values strongly correlate with fraudulent activity
-   High-risk geographic regions show elevated fraud probability

------------------------------------------------------------------------

## ⚙️ Setup Instructions

### 1️⃣ Environment Setup

``` bash
conda create -n fraud-detection python=3.9
conda activate fraud-detection
pip install -r requirements.txt
```

### 2️⃣ Run Full Pipeline

``` bash
python src/pipeline.py     --ecom_data data/raw/Fraud_Data.csv     --cc_data data/raw/creditcard.csv
```

### 3️⃣ Generate Reports

``` bash
jupyter nbconvert --to html notebooks/*.ipynb --output-dir reports/
```

------------------------------------------------------------------------

## 📦 Dependencies

-   Python 3.9+
-   pandas, numpy, scikit-learn
-   xgboost==1.6.2
-   imbalanced-learn
-   matplotlib, shap
-   jupyter, nbconvert

------------------------------------------------------------------------

## 📈 Business Impact

-   Reduced financial losses due to fraud
-   Improved transaction risk scoring
-   Transparent AI decisions with SHAP explainability
-   Production-ready deployment architecture

------------------------------------------------------------------------

## 📄 Final Report

-   Download PDF Report\
-   View Blog Post

------------------------------------------------------------------------

💡 Designed for production deployment and real-time fraud risk
monitoring.
