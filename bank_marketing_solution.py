"""
Task 1: Term Deposit Subscription Prediction
Bank Marketing Dataset - Complete Solution
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, roc_curve, auc, ConfusionMatrixDisplay
)
import shap

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD & EXPLORE DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING & EXPLORING DATASET")
print("=" * 60)

# Load the UCI Bank Marketing dataset (semicolon-separated)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

# Download and load
import urllib.request, zipfile, io

print("Downloading dataset from UCI...")
try:
    response = urllib.request.urlopen(url, timeout=15)
    z = zipfile.ZipFile(io.BytesIO(response.read()))
    df = pd.read_csv(z.open("bank-edited.csv"), sep=";")
    print(f"Dataset loaded successfully! Shape: {df.shape}")
except Exception as e:
    print(f"Download failed ({e}), generating realistic synthetic data instead...")
    # Synthetic dataset with REAL learnable patterns (not pure random noise)
    np.random.seed(42)
    n = 4521

    age       = np.random.randint(18, 75, n)
    job       = np.random.choice(['admin.','technician','services','management','retired',
                                   'blue-collar','unemployed','entrepreneur','housemaid',
                                   'unknown','self-employed','student'], n)
    marital   = np.random.choice(['married','single','divorced'], n)
    education = np.random.choice(['secondary','tertiary','primary','unknown'], n)
    default   = np.random.choice(['no','yes'], n, p=[0.98, 0.02])
    balance   = np.random.randint(-500, 10000, n)
    housing   = np.random.choice(['yes','no'], n, p=[0.56, 0.44])
    loan      = np.random.choice(['yes','no'], n, p=[0.16, 0.84])
    contact   = np.random.choice(['cellular','telephone','unknown'], n)
    day       = np.random.randint(1, 31, n)
    month     = np.random.choice(['jan','feb','mar','apr','may','jun',
                                   'jul','aug','sep','oct','nov','dec'], n)
    duration  = np.random.randint(0, 3000, n)
    campaign  = np.random.randint(1, 20, n)
    pdays     = np.random.choice([-1] + list(range(1, 400)), n)
    previous  = np.random.randint(0, 10, n)
    poutcome  = np.random.choice(['unknown','failure','success','other'], n)

    # Build TARGET with realistic signal
    # Subscribers tend to: longer call, higher balance, prior success, fewer campaigns
    score = (
        (duration > 500).astype(float) * 1.5 +
        (balance > 1500).astype(float) * 0.8 +
        (poutcome == 'success').astype(float) * 2.0 +
        (campaign <= 2).astype(float) * 0.6 +
        (education == 'tertiary').astype(float) * 0.4 +
        (contact == 'cellular').astype(float) * 0.3 +
        np.random.normal(0, 0.5, n)
    )
    prob_yes = 1 / (1 + np.exp(-(score - 2.0)))
    y_arr    = np.where(np.random.random(n) < prob_yes, 'yes', 'no')

    df = pd.DataFrame({
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month, 'duration': duration,
        'campaign': campaign, 'pdays': pdays, 'previous': previous,
        'poutcome': poutcome, 'y': y_arr
    })
    print(f"Realistic synthetic dataset created. Shape: {df.shape}")

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Target Distribution ---")
print(df['y'].value_counts())
print(f"Class balance: {df['y'].value_counts(normalize=True).round(3).to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: ENCODING CATEGORICAL FEATURES")
print("=" * 60)

"""
WHY DO WE NEED ENCODING?
─────────────────────────
ML algorithms (especially Logistic Regression) work with NUMBERS only.
Categorical columns like 'job', 'marital', 'education' contain text strings.

Two main encoding strategies:
  1. Label Encoding  → converts each category to an integer (0, 1, 2...)
                       ✅ Good for: binary columns (yes/no) or ordinal data
                       ❌ Bad for: nominal data — implies fake ordering

  2. One-Hot Encoding (pd.get_dummies) → creates a new binary column per category
                       ✅ Good for: nominal columns (no natural order)
                       ❌ Bad for: high cardinality (too many unique values)
"""

df_encoded = df.copy()

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('y')  # remove target
print(f"Categorical columns to encode: {categorical_cols}")

# Binary columns → Label Encoding (only 2 unique values)
binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
print(f"\nBinary cols (Label Encoded): {binary_cols}")

le = LabelEncoder()
for col in binary_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Multi-category columns → One-Hot Encoding
multi_cols = [col for col in categorical_cols if col not in binary_cols]
print(f"\nMulti-category cols (One-Hot Encoded): {multi_cols}")

df_encoded = pd.get_dummies(df_encoded, columns=multi_cols, drop_first=True)
print(f"Shape after encoding: {df_encoded.shape}")

# Encode target variable
df_encoded['y'] = (df_encoded['y'] == 'yes').astype(int)
print(f"\nTarget encoded: 'yes'=1, 'no'=0")
print(f"Final dataset shape: {df_encoded.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: TRAIN/TEST SPLIT & FEATURE SCALING")
print("=" * 60)

X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: TRAINING MODELS")
print("=" * 60)

# Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
print(f"  LR F1-Score: {f1_score(y_test, lr_preds):.4f}")

# Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train, y_train)     # RF doesn't need scaling
rf_preds = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
print(f"  RF F1-Score: {f1_score(y_test, rf_preds):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: EVALUATION — CONFUSION MATRIX, F1, ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: MODEL EVALUATION")
print("=" * 60)

print("\n--- Logistic Regression Classification Report ---")
print(classification_report(y_test, lr_preds, target_names=['No','Yes']))

print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, rf_preds, target_names=['No','Yes']))

# ── PLOT 1: Confusion Matrices ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices", fontsize=16, fontweight='bold', y=1.02)

for ax, preds, title in zip(axes,
                             [lr_preds, rf_preds],
                             ["Logistic Regression", "Random Forest"]):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No','Yes'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/01_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 01_confusion_matrices.png")

# ── PLOT 2: ROC Curves ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

for proba, label, color in [
    (lr_proba, "Logistic Regression", "#e74c3c"),
    (rf_proba, "Random Forest",       "#2ecc71"),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/02_roc_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 02_roc_curves.png")

# ── PLOT 3: F1 Score Comparison ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
models = ['Logistic Regression', 'Random Forest']
f1s    = [f1_score(y_test, lr_preds), f1_score(y_test, rf_preds)]
bars   = ax.bar(models, f1s, color=['#3498db', '#2ecc71'], width=0.4, edgecolor='black')
ax.set_ylim(0, 1)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("F1-Score Comparison", fontsize=14, fontweight='bold')
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.4f}", ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/03_f1_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 03_f1_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SHAP EXPLAINABILITY (5 Predictions)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: SHAP — EXPLAINING 5 MODEL PREDICTIONS")
print("=" * 60)

"""
WHAT IS SHAP?
─────────────
SHAP (SHapley Additive exPlanations) tells you WHY the model made a specific prediction.
For each prediction, it assigns each feature a "SHAP value":
  → Positive SHAP value = this feature PUSHED the prediction toward 'subscribed'
  → Negative SHAP value = this feature PUSHED the prediction away from 'subscribed'
"""

# Use Random Forest with TreeExplainer (fastest for tree-based models)
print("Computing SHAP values for Random Forest...")
explainer   = shap.TreeExplainer(rf_model)
X_test_df   = pd.DataFrame(X_test, columns=X.columns)

# Pick 5 diverse test samples
sample_indices = [0, 1, 2, 3, 4]
X_sample = X_test_df.iloc[sample_indices]
shap_values = explainer(X_sample)

# ── PLOT 4: SHAP Waterfall plots for 5 predictions ───────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(12, 28))
fig.suptitle("SHAP Waterfall Plots — Explaining 5 Individual Predictions\n(Random Forest)",
             fontsize=15, fontweight='bold', y=1.01)

for i, idx in enumerate(sample_indices):
    actual    = y_test.iloc[idx]
    predicted = rf_preds[idx]
    prob      = rf_proba[idx]

    plt.sca(axes[i])
    shap.plots.waterfall(shap_values[i, :, 1], max_display=8, show=False)
    axes[i].set_title(
        f"Sample {i+1} | Actual: {'Subscribed' if actual==1 else 'Not Subscribed'} | "
        f"Predicted: {'Subscribed' if predicted==1 else 'Not Subscribed'} | "
        f"Prob: {prob:.2f}",
        fontsize=10, fontweight='bold'
    )
    print(f"  Sample {i+1}: Actual={actual}, Predicted={predicted}, P(subscribe)={prob:.3f}")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/04_shap_waterfall_5_predictions.png",
            dpi=120, bbox_inches='tight')
plt.close()
print("✅ Saved: 04_shap_waterfall_5_predictions.png")

# ── PLOT 5: Global SHAP Summary (bonus — always useful to include) ────────────
print("\nComputing global SHAP values (this may take a moment)...")
shap_vals_all = explainer(X_test_df.head(200))

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_vals_all[:, :, 1].values, X_test_df.head(200),
                  plot_type="bar", show=False, max_display=15)
plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/05_shap_global_importance.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 05_shap_global_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)

print(f"""
┌─────────────────────────┬──────────────────┬───────────────┐
│ Metric                  │ Logistic Reg.    │ Random Forest │
├─────────────────────────┼──────────────────┼───────────────┤
│ F1-Score                │ {f1_score(y_test, lr_preds):.4f}           │ {f1_score(y_test, rf_preds):.4f}         │
│ ROC-AUC                 │ {auc(lr_fpr, lr_tpr):.4f}           │ {auc(rf_fpr, rf_tpr):.4f}         │
└─────────────────────────┴──────────────────┴───────────────┘

Output files:
  01_confusion_matrices.png
  02_roc_curves.png
  03_f1_comparison.png
  04_shap_waterfall_5_predictions.png
  05_shap_global_importance.png
""")

print("All done! ✅")
