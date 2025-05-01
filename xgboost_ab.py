import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score
)
from xgboost import XGBClassifier

# Load data
df = pd.read_excel("final_dataframe.xlsx")
results = []

# Antibiotic label columns
antibiotics = [
    'sefazolin', 'seftazidim', 'gentamisin', 'tmp_smx', 'sefiksim',
    'ertapenem', 'pip_tazo', 'amoxicillin_clavulanic_acid', 'seftriakson',
    'sefuroksim_aksetil', 'sefuroksim', 'ampisilin', 'nitrofurantoin',
    'siprofloksasin', 'fosfomisin', 'sefepim', 'amikasin', 'sefoksitin',
    'meropenem', 'sefotaksim'
]

# Feature columns
feature_columns = [col for col in df.columns if col not in antibiotics]

sns.set(style="whitegrid")

# Range of PCA components to try
pca_range = range(1, 50)

for ab in antibiotics:
    ab_df = df[feature_columns + [ab]].dropna()
    if len(ab_df[ab].unique()) < 2:
        print(f"Skipped {ab} due to only one class present.")
        continue

    total = len(ab_df)
    pos = ab_df[ab].sum()
    neg = total - pos

    X = ab_df[feature_columns]
    y = ab_df[ab]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_result = None

    for n_comp in pca_range:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp)),
            ('xgb', XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                max_depth=5,
                learning_rate=0.1,
                n_estimators=100,
                scale_pos_weight=neg / pos,
                random_state=42
            ))
        ])

        # Predict and evaluate
        try:
            y_pred = cross_val_predict(pipeline, X, y, cv=skf)
            y_proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]

            f1 = f1_score(y, y_pred)
            recall = recall_score(y, y_pred)
            precision = precision_score(y, y_pred)
            roc_auc_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')
            roc_auc_mean = roc_auc_scores.mean()
            roc_auc_std = roc_auc_scores.std()

            if best_result is None or roc_auc_mean > best_result["CV_ROC_AUC"]:
                best_result = {
                    "Antibiotic": ab,
                    "Samples": total,
                    "Num_0": neg,
                    "Num_1": pos,
                    "PCA_Components": n_comp,
                    "CV_ROC_AUC": round(roc_auc_mean, 3),
                    "ROC_AUC_STD": round(roc_auc_std, 3),
                    "F1_Score": round(f1, 3),
                    "Recall": round(recall, 3),
                    "Precision": round(precision, 3)
                }

        except Exception as e:
            print(f"Error for {ab} with {n_comp} components: {e}")
            continue

    if best_result:
        results.append(best_result)

# Results DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plot
results_df_sorted = results_df.sort_values(by="CV_ROC_AUC", ascending=True)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=results_df_sorted,
    x="CV_ROC_AUC",
    y="Antibiotic",
    palette="viridis"
)

plt.title("Best CV ROC AUC by Antibiotic (XGBoost with PCA Grid Search)", fontsize=16)
plt.xlabel("CV ROC AUC (Mean)", fontsize=14)
plt.ylabel("Antibiotic", fontsize=14)
plt.xlim(0.0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
