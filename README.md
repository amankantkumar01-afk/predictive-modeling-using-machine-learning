# Predictive Modeling Using Machine Learning

**Student:** AMAN KANT KUMAR  
**Field:** Data Science  
**Project:** Student Performance Prediction

## Objective
The aim of this project is to predict whether a student is likely to pass or fail using simple academic data.

## Dataset
The dataset contains:
- Study_Hours
- Attendance_Percent
- Previous_Score
- Passed (0 = Fail, 1 = Pass)

The dataset is included as `student_performance.csv`.

## Models Used
1. Decision Tree Classifier
2. Random Forest Classifier

The data was divided into 80% training data and 20% testing data.

## Evaluation
Accuracy, confusion matrix, classification report and ROC curve were used to check the model performance.

## How to Run
Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Then run:

```bash
python predictive_model.py
```

The program creates:
- confusion_matrix.png
- roc_curve.png
- feature_importance.png

## Conclusion
The models were able to learn a relationship between study hours, attendance and previous score and the final pass/fail result. Random Forest was also checked against the Decision Tree model. This project helped me understand the basic steps of supervised machine learning, from preparing data to training and evaluating a model.
