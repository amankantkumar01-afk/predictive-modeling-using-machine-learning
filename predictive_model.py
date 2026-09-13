# Predictive Modeling Using Machine Learning
# Student Performance Prediction
# Name: AMAN KANT KUMAR

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# 1. Load data
data = pd.read_csv("student_performance.csv")
print(data.head())
print(data.info())

# 2. Select input and output
X = data[["Study_Hours", "Attendance_Percent", "Previous_Score"]]
y = data["Passed"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
print("\nDecision Tree Accuracy:", round(dt_accuracy, 3))
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# 5. Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
print("\nRandom Forest Accuracy:", round(rf_accuracy, 3))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# 6. Confusion matrix for Random Forest
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xticks([0,1], ["Fail", "Pass"])
plt.yticks([0,1], ["Fail", "Pass"])
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# 7. ROC curve
fpr, tpr, thresholds = roc_curve(y_test, rf_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()

# 8. Simple feature importance
importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.plot(kind="bar", title="Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nProject completed successfully.")
