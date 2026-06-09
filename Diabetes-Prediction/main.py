import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading dataset
import os

print("Current Folder:")
print(os.getcwd())

print("\nFiles in Folder:")
print(os.listdir())

diabetes_dataset = pd.read_csv(r'Diabtes Prediction\diabetes.csv',sep='\t')
print(diabetes_dataset.columns)
print(diabetes_dataset.shape)

# Separating features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardization
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Model Training
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy on train data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(
    X_train_prediction,
    Y_train
)

print("Training Accuracy:", training_data_accuracy)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(
    X_test_prediction,
    Y_test
)

print("Test Accuracy:", test_data_accuracy)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading dataset
print(diabetes_dataset.columns)
print(diabetes_dataset.shape)

# Separating features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardization
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Model Training
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy on train data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(
    X_train_prediction,
    Y_train
)

print("Training Accuracy:", training_data_accuracy)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(
    X_test_prediction,
    Y_test
)

print("Test Accuracy:", test_data_accuracy)

import tkinter as tk
from tkinter import messagebox

def predict_diabetes():
    try:
        pregnancies = float(preg_entry.get())
        glucose = float(glucose_entry.get())
        bp = float(bp_entry.get())
        skin = float(skin_entry.get())
        insulin = float(insulin_entry.get())
        bmi = float(bmi_entry.get())
        dpf = float(dpf_entry.get())
        age = float(age_entry.get())

        input_df = pd.DataFrame(
            [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
            columns=[
                'Pregnancies',
                'Glucose',
                'BloodPressure',
                'SkinThickness',
                'Insulin',
                'BMI',
                'DiabetesPedigreeFunction',
                'Age'
            ]
        )

        input_scaled = scaler.transform(input_df)

        prediction = classifier.predict(input_scaled)

        if prediction[0] == 0:
            result_label.config(
                text="✅ Person is NOT Diabetic",
                fg="green"
            )
        else:
            result_label.config(
                text="⚠️ Person IS Diabetic",
                fg="red"
            )

    except:
        messagebox.showerror(
            "Error",
            "Please enter valid values"
        )

# GUI Window
root = tk.Tk()
root.title("Diabetes Predictor")
root.geometry("500x650")

tk.Label(
    root,
    text="🩺 Diabetes Prediction System",
    font=("Arial", 20, "bold")
).pack(pady=15)

# Pregnancies
tk.Label(root, text="Pregnancies").pack()
preg_entry = tk.Entry(root)
preg_entry.pack()

# Glucose
tk.Label(root, text="Glucose").pack()
glucose_entry = tk.Entry(root)
glucose_entry.pack()

# Blood Pressure
tk.Label(root, text="Blood Pressure").pack()
bp_entry = tk.Entry(root)
bp_entry.pack()

# Skin Thickness
tk.Label(root, text="Skin Thickness").pack()
skin_entry = tk.Entry(root)
skin_entry.pack()

# Insulin
tk.Label(root, text="Insulin").pack()
insulin_entry = tk.Entry(root)
insulin_entry.pack()

# BMI
tk.Label(root, text="BMI").pack()
bmi_entry = tk.Entry(root)
bmi_entry.pack()

# Diabetes Pedigree Function
tk.Label(root, text="Diabetes Pedigree Function").pack()
dpf_entry = tk.Entry(root)
dpf_entry.pack()

# Age
tk.Label(root, text="Age").pack()
age_entry = tk.Entry(root)
age_entry.pack()

tk.Button(
    root,
    text="Predict",
    command=predict_diabetes,
    bg="green",
    fg="white",
    width=20
).pack(pady=15)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14, "bold")
)
result_label.pack(pady=20)

root.mainloop()
