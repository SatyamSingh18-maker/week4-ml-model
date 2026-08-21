import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# -----------------------------
# 1. DATA LOADING
# -----------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = malignant, 1 = benign

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Class balance:\n", df["target"].value_counts())
print("Target names:", data.target_names)

# -----------------------------
# 2. DATA PREPARATION
# -----------------------------
X = df.drop(columns=["target"])
y = df["target"]

# Train/test split, stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Feature scaling — required for logistic regression to converge properly
# and to prevent large-magnitude features from dominating the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -----------------------------
# 3. MODEL SELECTION & TRAINING
# -----------------------------
# Logistic Regression chosen because:
# - Binary classification target (malignant vs benign)
# - Interpretable coefficients (important in a medical context)
# - Strong baseline before trying more complex models
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# 4. EVALUATION METRICS
# -----------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== TEST SET METRICS ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 score:  {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# Cross-validation to check for overfitting / model stability
cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring="accuracy")
print(f"\n5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print("Individual fold scores:", cv_scores)

# Train accuracy (to compare vs test accuracy — check for overfitting)
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
print(f"\nTrain accuracy: {train_acc:.4f} vs Test accuracy: {acc:.4f}")
print(f"Gap: {train_acc - acc:.4f} (small gap suggests low overfitting)")

# -----------------------------
# 5. VISUALIZATIONS
# -----------------------------

# Viz 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix — Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("viz1_confusion_matrix.png")
plt.close()

# Viz 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6.5, 5.5))
plt.plot(fpr, tpr, color="#2E86AB", linewidth=2.5, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("viz2_roc_curve.png")
plt.close()

# Viz 3: Precision-Recall curve
prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6.5, 5.5))
plt.plot(rec_arr, prec_arr, color="#A23B72", linewidth=2.5)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("viz3_precision_recall.png")
plt.close()

# Viz 4: Top 10 feature coefficients (interpretability)
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False).head(10)
plt.figure(figsize=(8, 6))
colors = ["#C0392B" if c < 0 else "#27AE60" for c in coef_df["coefficient"]]
plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
plt.xlabel("Coefficient value (standardized features)")
plt.title("Top 10 Most Influential Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("viz4_feature_importance.png")
plt.close()

# Viz 5: Cross-validation fold scores
plt.figure(figsize=(7, 5))
plt.bar(range(1, 6), cv_scores, color="#5DADE2")
plt.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean = {cv_scores.mean():.3f}")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("5-Fold Cross-Validation Accuracy")
plt.ylim(0.85, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig("viz5_cv_scores.png")
plt.close()

print("\nAll visualizations saved.")
