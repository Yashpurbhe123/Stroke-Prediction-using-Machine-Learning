# 🧠 Stroke Prediction Using Advanced Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Random%20Forest-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**A comprehensive machine learning pipeline for predicting stroke risk using healthcare data**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Documentation](#-documentation)

</div>

---
## 🎯 Overview

Stroke is a leading cause of death and disability worldwide. Early prediction and prevention can save lives. This project implements a **state-of-the-art machine learning pipeline** to predict stroke risk based on various health parameters and lifestyle factors.

### Key Highlights

- ✅ **High Accuracy**: Achieved 93.47% accuracy and 98.61% ROC-AUC with XGBoost
- 📊 **12 Professional Visualizations**: Comprehensive data analysis and model insights
- ⚖️ **Handles Imbalanced Data**: SMOTE implementation for balanced predictions
- 🔍 **Feature Importance Analysis**: Identifies key risk factors for stroke
- 📈 **Cross-Validation**: Robust 5-fold stratified cross-validation (98.22% mean)
- 🎨 **Production-Ready Code**: Clean, documented, and industry-standard practices
- 📓 **Interactive Analysis**: Complete Jupyter Notebook implementation

---

## ✨ Features

### 🔬 Advanced Analytics
- Comprehensive Exploratory Data Analysis (EDA)
- Statistical testing and correlation analysis
- Bivariate and multivariate analysis
- Outlier detection and treatment

### 🤖 Machine Learning Models
- **XGBoost Classifier** - Gradient boosting with hyperparameter tuning
- **Random Forest Classifier** - Ensemble learning with optimized parameters
- SMOTE for handling class imbalance
- Feature scaling and preprocessing

### 📊 Visualizations
- Missing values analysis
- Target distribution and class imbalance visualization
- Feature distributions with KDE overlays
- Correlation heatmaps
- ROC curves and AUC scores
- Precision-Recall curves
- Confusion matrices
- Feature importance plots (Gain & Weight)
- Learning curves
- Cross-validation performance

### 🎯 Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix Analysis
- Classification Reports
- Cross-Validation Scores

---

## 📊 Dataset

### Dataset Characteristics
- **Total Records**: 5,110 patients
- **Features**: 11 clinical and demographic features
- **Target Variable**: Stroke (Binary: 0 = No Stroke, 1 = Stroke)
- **Class Imbalance**: ~95% No Stroke, ~5% Stroke

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| `id` | Numeric | Unique patient identifier |
| `gender` | Categorical | Male, Female, Other |
| `age` | Numeric | Age of the patient |
| `hypertension` | Binary | 0 = No hypertension, 1 = Has hypertension |
| `heart_disease` | Binary | 0 = No heart disease, 1 = Has heart disease |
| `ever_married` | Categorical | Yes or No |
| `work_type` | Categorical | Children, Govt_job, Never_worked, Private, Self-employed |
| `Residence_type` | Categorical | Rural or Urban |
| `avg_glucose_level` | Numeric | Average glucose level in blood |
| `bmi` | Numeric | Body Mass Index |
| `smoking_status` | Categorical | Formerly smoked, Never smoked, Smokes, Unknown |
| `stroke` | Binary | 0 = No stroke, 1 = Had stroke (Target) |

---

## 🚀 Installation

### Clone the Repository
```bash
git clone https://github.com/Yashpurbhe123/Stroke-Prediction-using-Machine-Learning.git
cd Stroke-Prediction-using-Machine-Learning
```

### Install Dependencies
```bash
pip install -r requirements.txt
```


### Running the Notebook

1. **Place your dataset** in the project root directory:
```bash
healthcare-dataset-stroke-data.csv
```

2. **Launch Jupyter Notebook**:
```bash
jupyter notebook
```

3. **Open the notebook**:
   - Navigate to `stroke_prediction.ipynb`
   - Run all cells sequentially (Cell → Run All)

4. **View results**:
   - Interactive output in notebook cells
   - 12 high-resolution PNG visualizations saved automatically
   - Model comparison and performance summary displayed inline


## 🔬 Methodology

### 1️⃣ Data Preprocessing
- **Missing Value Treatment**: Median imputation for BMI (201 missing values)
- **Outlier Analysis**: IQR-based detection for BMI and glucose levels
- **Feature Engineering**: One-hot encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features

### 2️⃣ Handling Class Imbalance
- **SMOTE (Synthetic Minority Over-sampling Technique)**
  - Original: 4,861 No Stroke | 249 Stroke (19.5:1 ratio)
  - After SMOTE: Balanced 50:50 distribution
  - K-neighbors = 5 for synthetic sample generation

### 3️⃣ Model Training
- **Train-Test Split**: 80:20 stratified split
- **Models**: XGBoost & Random Forest
- **Hyperparameters**:
  ```python
  XGBoost:
    - n_estimators: 200
    - max_depth: 6
    - learning_rate: 0.1
    - subsample: 0.8
    
  Random Forest:
    - n_estimators: 200
    - max_depth: 10
    - min_samples_split: 5
  ```

### 4️⃣ Model Evaluation
- **5-Fold Stratified Cross-Validation**
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: ROC curves, PR curves, confusion matrices

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV Mean | CV Std |
|-------|----------|-----------|--------|----------|---------|---------|---------|
| **XGBoost** | **0.9347** | **0.9179** | **0.9547** | **0.9360** | **0.9861** | **0.9822** | **0.0020** |
| **Random Forest** | 0.8920 | 0.8608 | 0.9352 | 0.8964 | 0.9618 | 0.9560 | 0.0030 |

### Key Insights

🎯 **Best Model**: XGBoost Classifier
- **Highest ROC-AUC**: 0.9861 (98.61%)
- **Excellent Accuracy**: 93.47%
- **Strong Precision & Recall Balance**: 91.79% precision, 95.47% recall
- **Robust Cross-Validation**: 98.22% ± 0.20%

### XGBoost Classification Report
```
              precision    recall  f1-score   support

   No Stroke       0.95      0.91      0.93       973
      Stroke       0.92      0.95      0.94       972

    accuracy                           0.93      1945
   macro avg       0.94      0.93      0.93      1945
weighted avg       0.94      0.93      0.93      1945
```

### Random Forest Classification Report
```
              precision    recall  f1-score   support

   No Stroke       0.93      0.85      0.89       973
      Stroke       0.86      0.94      0.90       972

    accuracy                           0.89      1945
   macro avg       0.89      0.89      0.89      1945
weighted avg       0.89      0.89      0.89      1945
```

🔍 **Top Risk Factors** (Feature Importance):
1. **Age** - Strongest predictor of stroke risk
2. **Average Glucose Level** - Second most important factor
3. **BMI** - Significant contributor
4. **Hypertension** - Notable risk factor
5. **Heart Disease** - Important comorbidity

📊 **Clinical Implications**:
- Age > 60 significantly increases stroke risk
- Glucose levels > 200 mg/dL require immediate attention
- BMI in obese range (>30) correlates with higher risk
- Model achieves 95% recall on stroke cases (catches 95% of actual strokes)

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms

### Machine Learning
- **XGBoost** - Gradient boosting framework
- **Random Forest** - Ensemble learning
- **Imbalanced-learn** - SMOTE implementation

### Visualization
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualization
- **Plotly** - Interactive visualizations

### Statistical Analysis
- **SciPy** - Scientific computing
- **Statsmodels** - Statistical tests

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ for healthcare and AI**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=Yashpurbhe123.stroke-prediction-ml)

</div>
