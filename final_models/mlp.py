import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

# Load Data
#df = pd.read_excel("final_dataframe_cleaned.xlsx")
df= pd.read_csv("final_dataframe_with_clinicalBERT_PCA.csv")
results = []

# Drop unwanted antibiotic labels
df = df.drop(columns=['sefazolin', 'fosfomisin', 'meropenem'])

# Antibiotic labels
antibiotics = ['seftazidim', 'gentamisin', 'tmp_smx', 'sefiksim',
    'ertapenem', 'pip_tazo', 'amoxicillin_clavulanic_acid', 'seftriakson',
    'sefuroksim_aksetil', 'sefuroksim', 'ampisilin', 'nitrofurantoin',
    'siprofloksasin', 'sefepim', 'amikasin', 'sefoksitin',
    'sefotaksim'
]

# Feature columns
feature_columns = [col for col in df.columns if col not in antibiotics]

# Set Seaborn style
sns.set(style="whitegrid")

# Loop over antibiotics
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

    pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.99)),
    ('oversample', RandomOverSampler(random_state=42)),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True
    ))
    ])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Predictions
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    y_proba = cross_val_predict(pipeline, X, y, cv=skf, method='predict_proba')[:, 1]

    # Metrics
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)

    # Store results
    results.append({
        "Antibiotic": ab,
        "Samples": total,
        "Num_0": neg,
        "Num_1": pos,
        "F1_Score": round(f1, 3),
        "Recall": round(recall, 3),
        "Precision": round(precision, 3)
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Sort by F1 Score
results_df_sorted = results_df.sort_values(by="F1_Score", ascending=False)
print(results_df_sorted)

# Plot F1 Score
plt.figure(figsize=(12, 8))
sns.barplot(
    data=results_df_sorted,
    x="F1_Score",
    y="Antibiotic",
    palette="coolwarm"
)

plt.title("F1 Score by Antibiotic (MLPClassifier with PCA)", fontsize=16)
plt.xlabel("F1 Score", fontsize=14)
plt.ylabel("Antibiotic", fontsize=14)
plt.xlim(0.0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
