import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading dataset
diabetes_dataset = pd.read_csv('diabetes.csv', sep='\t')
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

# Prediction System
input_data = (4,110,92,0,0,37.6,0.191,30)

input_data_df = pd.DataFrame(
    [input_data],
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

std_data = scaler.transform(input_data_df)

prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
