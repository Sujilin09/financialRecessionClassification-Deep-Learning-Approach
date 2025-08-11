# Financial Crisis Classification using Macroeconomic Indicators – LSTM

A Deep Learning Approach for Binary Recession Classification  
Course: MDS471 - Neural Network and Deep Learning  
 
August 2025


## Project Overview

Financial crises disrupt economies worldwide, yet early warning signals exist within complex macroeconomic data. This project leverages Long Short-Term Memory (LSTM) neural networks to classify and forecast recessions six months ahead using macroeconomic indicators.

The goal is to detect early signals of financial crises from indicators such as Industrial Production, Retail Sales, Consumer Sentiment, Investment, and the VIX, enabling proactive decision-making for governments, investors, and businesses.

---

## Key Features

- Utilizes a comprehensive dataset of monthly macroeconomic indicators (2000-2023) combined with the US Recession Indicator (USRECD).
- Applies extensive data preprocessing including missing value imputation, normalization, and sequence creation.
- Addresses class imbalance using SMOTE oversampling and class-weighted focal loss.
- Employs feature selection via correlation analysis to identify top 10 recession-relevant indicators.
- Implements a two-layer LSTM deep learning architecture with dropout for robust time series classification.
- Forecasts recession status 6 months ahead based on 12 months of historical data.
- Evaluates model performance with Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.

---
##  LSTM Architecture – Recession Prediction Model

This model is a **two-layer Long Short-Term Memory (LSTM)** network designed to capture temporal patterns in macroeconomic indicators and **forecast recessions 6 months ahead**.

---

### Input Layer
- **Sequence length:** 12 months
- **Features:** 10 normalized macroeconomic indicators

---

###  LSTM Layers

#### LSTM Layer 1
- Units: **64**
- Activation: `tanh`
- Dropout: `0.3` (on input connections)
- Recurrent Dropout: `0.3` (on recurrent connections)
- **Purpose:** Capture complex temporal dependencies and filter noise.

#### LSTM Layer 2
- Units: **32**
- Activation: `tanh`
- Dropout: `0.3`
- Recurrent Dropout: `0.3`
- **Purpose:** Further refine temporal features and enhance generalization.

---

###  Output Layer
- Dense layer with **1 neuron**
- Activation: `sigmoid`
- **Output:** Probability of recession (`1 = recession`, `0 = non-recession`)

---

### ⚙ Training Specifications
- **Optimizer:** Adam (`learning_rate = 0.001`)
- **Loss Function:** Focal Loss *(handles class imbalance)*
- **Class Balancing:** SMOTE oversampling applied on training data
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

---
##  Pipeline – Financial Crisis Classification with LSTM

The complete process for **financial crisis classification** using the LSTM model involves the following key steps:

---

###  Data Collection
- Gather **monthly macroeconomic indicator data** spanning **2000–2023**.
- Obtain the **US Recession Indicator (USRECD)** as the binary target variable from **FRED**.

---

###  Data Cleaning & Preprocessing
- Convert **date formats** to a consistent standard for merging.
- Remove invalid or malformed records.
- Handle missing values using **forward-fill** to maintain temporal continuity.
- Normalize all features using **MinMaxScaler** to scale values between 0 and 1.

---

###  Feature Selection
- Calculate **Pearson correlation coefficients** between features and the recession target.
- Select the **top 10 features** most correlated with recession periods for model input.

---

###  Target Shifting
- Shift recession labels **6 months ahead** to enable **forecasting** future recessions based on current and past data.

---

###  Sequencing
- Transform the time series into **12-month sequences** per sample, providing the model with **one year of historical context** for each prediction.

---

###  Class Imbalance Handling
- Apply **SMOTE** (Synthetic Minority Oversampling Technique) to oversample the minority recession class in training data.
- Use **Focal Loss** during model training to focus learning on **hard-to-classify recession samples**.

---

###  Model Training
- Train the **two-layer LSTM network** with:
  - Dropout regularization  
  - Adam optimizer  
  - Focal loss  
- Learn **temporal dependencies** and improve generalization.

---

###  Model Evaluation
- Evaluate using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion matrix  
  - ROC-AUC  
- Provides a **comprehensive performance assessment**, especially for imbalanced data.

---

###  Prediction & Deployment
- Use the trained LSTM model to **predict recession probabilities** on new, unseen macroeconomic data.
- Support **proactive economic decision-making**.

---

## Usage

- **Data Preparation:**  
Run data preprocessing scripts to merge datasets, clean missing values, normalize features, and generate sequences.

- **Training:**  
Execute the training script to build and train the LSTM model with class balancing using SMOTE and focal loss.

- **Evaluation:**  
Evaluate the trained model using test data to generate performance metrics and ROC curve.

- **Prediction:**  
Use the trained model to predict recession probabilities on new macroeconomic data.

---

## Results Summary

- Model Accuracy: ~99%  
- ROC-AUC Score: 1.0 (near perfect classification)  
- Confusion Matrix: Low false positives (2) and false negatives (7), indicating strong classification capability.

The LSTM model effectively captures temporal dependencies in macroeconomic data, allowing for accurate early recession forecasting.

---

## Future Work

- Incorporate additional macroeconomic variables and alternative data sources for improved coverage.
- Implement interpretability methods (e.g., SHAP, LIME) to explain model decisions.
- Develop a real-time recession warning dashboard for practical deployment.

---




