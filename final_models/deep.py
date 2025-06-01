import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import torch

# Load data
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

sns.set(style="whitegrid")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=feature_columns)
    X_df['label'] = y.values

    # Define feature columns
    fixlen_feature_columns = [DenseFeat(feat, 1,) for feat in feature_columns]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []

    for train_idx, test_idx in skf.split(X_df, X_df['label']):
        train_data = X_df.iloc[train_idx]
        test_data = X_df.iloc[test_idx]

        train_model_input = {name: train_data[name].values for name in feature_names}
        test_model_input = {name: test_data[name].values for name in feature_names}

        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
        model.compile("adam", "binary_crossentropy", metrics=["auc"], )

        model.fit(train_model_input, train_data['label'].values, batch_size=64, epochs=10, verbose=0)

        pred_ans = model.predict(test_model_input, batch_size=256)
        pred_label = (pred_ans > 0.5).astype(int)

        f1 = f1_score(test_data['label'], pred_label)
        recall = recall_score(test_data['label'], pred_label)
        precision = precision_score(test_data['label'], pred_label)
        auc = roc_auc_score(test_data['label'], pred_ans)

        metrics.append((auc, f1, recall, precision))

    aucs, f1s, recalls, precisions = zip(*metrics)

    results.append({
        "Antibiotic": ab,
        "Samples": total,
        "Num_0": neg,
        "Num_1": pos,
        "CV_ROC_AUC": round(np.mean(aucs), 3),
        "ROC_AUC_STD": round(np.std(aucs), 3),
        "F1_Score": round(np.mean(f1s), 3),
        "Recall": round(np.mean(recalls), 3),
        "Precision": round(np.mean(precisions), 3)
    })

# Results
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
    palette="magma"
)

plt.title("F1 Score by Antibiotic (DeepFM)", fontsize=16)
plt.xlabel("F1 Score", fontsize=14)
plt.ylabel("Antibiotic", fontsize=14)
plt.xlim(0.0, 1.0)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()