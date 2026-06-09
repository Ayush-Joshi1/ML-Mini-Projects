import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
import os

print(os.getcwd())
student_data = pd.read_csv(
    r'Student Marks Predictor\student_data.csv'
)
print(student_data.head())

# Features and Target
X = student_data.drop(columns='Marks', axis=1)
Y = student_data['Marks']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, Y_train)

# Predictions
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

# Scores
train_score = r2_score(Y_train, train_prediction)
test_score = r2_score(Y_test, test_prediction)

print("\nTraining Score:", train_score)
print("Test Score:", test_score)

# New Student Data
import tkinter as tk
from tkinter import messagebox


def predict_marks():
    try:
        study_hours = float(hours_entry.get())
        attendance = float(attendance_entry.get())
        assignments = float(assignments_entry.get())

        input_data = [[study_hours, attendance, assignments]]

        prediction = model.predict(input_data)

        result_label.config(
            text=f"Predicted Marks: {round(prediction[0], 2)}"
        )

    except:
        messagebox.showerror(
            "Error",
            "Please enter valid numbers!"
        )


# ---------------- GUI ----------------

root = tk.Tk()
root.title("Student Marks Predictor")
root.geometry("500x450")

title = tk.Label(
    root,
    text="🎓 Student Marks Predictor",
    font=("Arial", 20, "bold")
)
title.pack(pady=20)

# Study Hours
tk.Label(root, text="Study Hours").pack()
hours_entry = tk.Entry(root)
hours_entry.pack(pady=5)

# Attendance
tk.Label(root, text="Attendance (%)").pack()
attendance_entry = tk.Entry(root)
attendance_entry.pack(pady=5)

# Assignments
tk.Label(root, text="Assignments Completed").pack()
assignments_entry = tk.Entry(root)
assignments_entry.pack(pady=5)

# Button
predict_btn = tk.Button(
    root,
    text="Predict Marks",
    command=predict_marks,
    bg="green",
    fg="white",
    width=20
)
predict_btn.pack(pady=20)

# Result
result_label = tk.Label(
    root,
    text="",
    font=("Arial", 16, "bold")
)
result_label.pack(pady=20)

root.mainloop()
