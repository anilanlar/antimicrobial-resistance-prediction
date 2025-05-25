import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Load Data
df = pd.read_excel("final_dataframe_ohe.xlsx")
results = []

# List of antibiotic label columns
antibiotics = [
    'sefazolin', 'seftazidim', 'gentamisin', 'tmp_smx', 'sefiksim',
    'ertapenem', 'pip_tazo', 'amoxicillin_clavulanic_acid', 'seftriakson',
    'sefuroksim_aksetil', 'sefuroksim', 'ampisilin', 'nitrofurantoin',
    'siprofloksasin', 'fosfomisin', 'sefepim', 'amikasin', 'sefoksitin',
    'meropenem', 'sefotaksim'
]

# Features are everything else
feature_columns = [col for col in df.columns if col not in antibiotics]

# Style for plots
sns.set(style="whitegrid")

# Loop over antibiotics
for ab in antibiotics:
    # Drop rows with nulls in any feature or the current antibiotic column
    ab_df = df[feature_columns + [ab]].dropna()

    # Skip if only one class present
    if len(ab_df[ab].unique()) < 2:
        print(f"Skipped {ab} due to only one class present.")
        continue

    # Sample info
    total = len(ab_df)
    pos = ab_df[ab].sum()
    neg = total - pos

    X = ab_df[feature_columns]
    y = ab_df[ab]

    # Model pipeline with scaling, PCA, and logistic regression
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.99),
        LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=skf)
    y_proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]

    # Metrics
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    roc_auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    roc_auc_mean = roc_auc_scores.mean()
    roc_auc_std = roc_auc_scores.std()

    # Save results
    results.append({
        "Antibiotic": ab,
        "Samples": total,
        "Num_0": neg,
        "Num_1": pos,
        "CV_ROC_AUC": round(roc_auc_mean, 3),
        "ROC_AUC_STD": round(roc_auc_std, 3),
        "F1_Score": round(f1, 3),
        "Recall": round(recall, 3),
        "Precision": round(precision, 3)
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Sort by ROC AUC for better visualization
results_df_sorted = results_df.sort_values(by="CV_ROC_AUC", ascending=True)

# Plot
plt.figure(figsize=(10, 8))
sns.barplot(
    data=results_df_sorted,
    x="CV_ROC_AUC",
    y="Antibiotic",
    palette="viridis"
)

plt.title("Mean CV ROC AUC by Antibiotic", fontsize=14)
plt.xlabel("CV ROC AUC (Mean)")
plt.ylabel("Antibiotic")
plt.xlim(0.0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
