import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
house_data = pd.read_csv(r'House Price Prediction/house_data.csv',sep='\t')

print(house_data.head())

# Features and Target
X = house_data.drop(columns='Price', axis=1)
Y = house_data['Price']

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Model
model = LinearRegression()

# Training
model.fit(X_train, Y_train)

# Prediction
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

# Accuracy
train_score = r2_score(Y_train, train_prediction)
test_score = r2_score(Y_test, test_prediction)

print("\nTraining Score:", train_score)
print("Test Score:", test_score)

import tkinter as tk
from tkinter import messagebox


def predict_price():
    try:
        area = float(area_entry.get())
        bedrooms = int(bedroom_entry.get())
        age = int(age_entry.get())

        input_data = pd.DataFrame(
            [[area, bedrooms, age]],
            columns=['Area', 'Bedrooms', 'Age']
        )

        prediction = model.predict(input_data)

        result_label.config(
            text=f"Predicted Price: ₹ {prediction[0]:,.0f}"
        )

    except:
        messagebox.showerror(
            "Error",
            "Please enter valid values"
        )


# GUI Window
root = tk.Tk()
root.title("House Price Predictor")
root.geometry("500x400")

title = tk.Label(
    root,
    text="🏠 House Price Predictor",
    font=("Arial", 18, "bold")
)
title.pack(pady=15)

# Area
tk.Label(root, text="Area (sq ft)").pack()
area_entry = tk.Entry(root)
area_entry.pack(pady=5)

# Bedrooms
tk.Label(root, text="Bedrooms").pack()
bedroom_entry = tk.Entry(root)
bedroom_entry.pack(pady=5)

# Age
tk.Label(root, text="House Age (years)").pack()
age_entry = tk.Entry(root)
age_entry.pack(pady=5)

# Predict Button
predict_btn = tk.Button(
    root,
    text="Predict Price",
    command=predict_price,
    bg="green",
    fg="white",
    width=20
)
predict_btn.pack(pady=15)

# Result Label
result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14, "bold")
)
result_label.pack(pady=10)

root.mainloop()
