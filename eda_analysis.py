import pandas as pd
import matplotlib.pyplot as plt

# Exploratory Data Analysis - Student Performance
df = pd.read_csv("student_performance_eda.csv")

print("----- BASIC INFORMATION -----")
print("Rows and columns:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\n----- STATISTICAL SUMMARY -----")
print(df.describe())

# Simple cleaning for analysis
df["Study_Hours"] = df["Study_Hours"].fillna(df["Study_Hours"].median())
df["Attendance_Percent"] = df["Attendance_Percent"].fillna(df["Attendance_Percent"].median())
df["Previous_Score"] = df["Previous_Score"].fillna(df["Previous_Score"].median())
df = df.drop_duplicates().reset_index(drop=True)

print("\nRows after cleaning:", len(df))

print("\n----- CORRELATION -----")
print(df[["Study_Hours", "Attendance_Percent", "Previous_Score"]].corr())

print("\n----- PASS COUNTS -----")
print(df["Passed"].value_counts())

print("\nEDA completed. Charts are included in the project folder.")
