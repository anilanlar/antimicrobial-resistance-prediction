import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier 
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

# DeepFM imports
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import torch

# Config
sns.set(style="whitegrid")

datasets = {
    "Label Encodings": pd.read_excel("final_dataframe_cleaned.xlsx"),
    "Embeddings": pd.read_csv("final_dataframe_with_clinicalBERT_PCA.csv")
}

drop_cols = ['sefazolin', 'fosfomisin', 'meropenem']
antibiotics = [
    'seftazidim', 'gentamisin', 'tmp_smx', 'sefiksim',
    'ertapenem', 'pip_tazo', 'amoxicillin_clavulanic_acid', 'seftriakson',
    'sefuroksim_aksetil', 'sefuroksim', 'ampisilin', 'nitrofurantoin',
    'siprofloksasin', 'sefepim', 'amikasin', 'sefoksitin', 'sefotaksim'
]

all_results = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for dataset_name, df in datasets.items():
    df = df.drop(columns=drop_cols)
    feature_columns = [col for col in df.columns if col not in antibiotics]

    for model_name in ["LogisticRegression", "RandomForest", "XGBoost", "MLP", "DeepFM"]:
        for ab in antibiotics:
            ab_df = df[feature_columns + [ab]].dropna()

            if len(ab_df[ab].unique()) < 2:
                print(f"Skipped {ab} in {model_name} due to only one class present.")
                continue

            total = len(ab_df)
            pos = ab_df[ab].sum()
            neg = total - pos

            X = ab_df[feature_columns]
            y = ab_df[ab]

            if model_name == "DeepFM":
                # Standardize features (DeepFM needs scaled dense features)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

                # Define DeepFM features
                fixlen_feature_columns = [DenseFeat(feat, 1) for feat in feature_columns]
                dnn_feature_columns = fixlen_feature_columns
                linear_feature_columns = fixlen_feature_columns
                feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                metrics = []

                for train_idx, test_idx in skf.split(X_scaled_df, y):
                    train_data = X_scaled_df.iloc[train_idx]
                    test_data = X_scaled_df.iloc[test_idx]

                    train_model_input = {name: train_data[name].values for name in feature_names}
                    test_model_input = {name: test_data[name].values for name in feature_names}

                    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
                    model.compile("adam", "binary_crossentropy", metrics=["auc"])

                    model.fit(train_model_input, y.iloc[train_idx].values, batch_size=64, epochs=10, verbose=0)

                    pred_ans = model.predict(test_model_input, batch_size=256)
                    pred_label = (pred_ans > 0.5).astype(int)

                    f1 = f1_score(y.iloc[test_idx], pred_label)
                    recall = recall_score(y.iloc[test_idx], pred_label)
                    precision = precision_score(y.iloc[test_idx], pred_label)
                    auc = roc_auc_score(y.iloc[test_idx], pred_ans)

                    metrics.append((auc, f1, recall, precision))

                aucs, f1s, recalls, precisions = zip(*metrics)

                all_results.append({
                    "Antibiotic": ab,
                    "F1_Score": round(np.mean(f1s), 3),
                    "Recall": round(np.mean(recalls), 3),
                    "Precision": round(np.mean(precisions), 3),
                    "CV_ROC_AUC": round(np.mean(aucs), 3),
                    "ROC_AUC_STD": round(np.std(aucs), 3),
                    "Samples": total,
                    "Num_0": neg,
                    "Num_1": pos,
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Model_Dataset": f"{model_name} ({dataset_name})"
                })

            else:
                # Existing sklearn pipeline setup
                if model_name == "LogisticRegression":
                    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=0.99)),
                        ('clf', model)
                    ])
                elif model_name == "RandomForest":
                    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=0.99)),
                        ('clf', model)
                    ])
                elif model_name == "XGBoost":
                    model = XGBClassifier(
                        use_label_encoder=False,
                        eval_metric='logloss',
                        scale_pos_weight=neg / pos if pos != 0 else 1,
                        random_state=42,
                        n_estimators=100,
                        verbosity=0
                    )
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('pca', PCA(n_components=0.99)),
                        ('clf', model)
                    ])
                elif model_name == "MLP":
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

                y_pred = cross_val_predict(pipeline, X, y, cv=skf)
                y_proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]

                f1 = f1_score(y, y_pred)
                recall = recall_score(y, y_pred)
                precision = precision_score(y, y_pred)
                roc_auc_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')
                roc_auc_mean = roc_auc_scores.mean()
                roc_auc_std = roc_auc_scores.std()

                all_results.append({
                    "Antibiotic": ab,
                    "F1_Score": round(f1, 3),
                    "Recall": round(recall, 3),
                    "Precision": round(precision, 3),
                    "CV_ROC_AUC": round(roc_auc_mean, 3),
                    "ROC_AUC_STD": round(roc_auc_std, 3),
                    "Samples": total,
                    "Num_0": neg,
                    "Num_1": pos,
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Model_Dataset": f"{model_name} ({dataset_name})"
                })

# Final combined results DataFrame
results_df = pd.DataFrame(all_results)

results_df.to_excel("model_results.xlsx", index=False)

# Ranking & plotting as you already have
results_df['Rank'] = results_df.groupby('Antibiotic')['F1_Score'].rank("dense", ascending=False)
top_results = results_df[results_df['Rank'] == 1].sort_values("F1_Score", ascending=False)

unique_combos = top_results['Model_Dataset'].unique()
palette = dict(zip(unique_combos, sns.color_palette("tab10", n_colors=len(unique_combos))))

plt.figure(figsize=(14, 10))
sns.barplot(
    data=top_results,
    x="Antibiotic",
    y="F1_Score",
    hue="Model_Dataset",
    dodge=False,
    palette=palette
)

plt.title("Best F1 Score per Antibiotic by Model and Dataset", fontsize=16)
plt.ylabel("F1 Score", fontsize=14)
plt.xlabel("Antibiotic", fontsize=14)
plt.ylim(0.0, 1.0)

plt.xticks(rotation=45, ha='right')  # rotate and align labels
plt.tight_layout(pad=2.0)            # add padding
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Model (Dataset)", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
