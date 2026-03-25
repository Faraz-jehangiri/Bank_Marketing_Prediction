# 🏦 Bank Marketing Prediction

> Predicting term deposit subscriptions using machine learning — Logistic Regression & Random Forest

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-explainability-teal?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-amber?style=flat-square)

---

## 📌 About

A bank runs phone-based marketing campaigns to get customers to subscribe to a term deposit. This project builds a classification model to predict which customers are likely to subscribe — helping the bank prioritise outreach and reduce wasted calls.

---

## 📊 Dataset

Bank Marketing Dataset from the **UCI Machine Learning Repository**.  
Contains **4,521 customer records** with demographic, financial, and campaign interaction features.

- **Target variable:** `y` — whether the customer subscribed (`yes` / `no`)
- **Class imbalance:** ~88% No, ~12% Yes → handled using `class_weight='balanced'`
- **Features:** 17 raw → 42 after encoding

---

## 🏆 Results

| Model | F1-Score | ROC-AUC |
|---|---|---|
| Random Forest | **0.76** | **0.70** |
| Logistic Regression | 0.63 | 0.69 |

---

## 📓 Notebook Walkthrough

| Step | Description |
|---|---|
| 1 | **Load & Explore** — shape, dtypes, missing values, class distribution plots |
| 2 | **Encode Categorical Features** — Label Encoding for binary cols, One-Hot Encoding for nominal cols |
| 3 | **Train/Test Split & Scaling** — 80/20 stratified split, StandardScaler for Logistic Regression |
| 4 | **Train Models** — Logistic Regression and Random Forest with balanced class weights |
| 5 | **Evaluate** — Confusion Matrix, Classification Report, F1-Score, ROC-AUC Curve |
| 6 | **SHAP Explainability** — Waterfall plots for 5 individual predictions + global feature importance |

---

## 🛠️ Tech Stack

- `pandas` `numpy` — data manipulation
- `scikit-learn` — ML models, preprocessing, evaluation
- `shap` — model explainability
- `matplotlib` `seaborn` — visualizations

---

## 🚀 Run Locally

```bash
pip install pandas numpy scikit-learn shap matplotlib seaborn jupyter
jupyter notebook bank_marketing_prediction.ipynb
```

> Place `bank-edited.csv` in the same folder as the notebook before running.

---

## 📁 Project Structure

```
Bank_Marketing_Prediction/
├── bank_marketing_prediction.ipynb   # main notebook
└── bank-edited.csv                   # dataset
```

---

## 👤 Author

**Faraz Jehangiri** — CS undergrad at SSUET Karachi, Data Analyst Intern at Developers Hub Corporation.  
[github.com/Faraz-jehangiri](https://github.com/Faraz-jehangiri)
