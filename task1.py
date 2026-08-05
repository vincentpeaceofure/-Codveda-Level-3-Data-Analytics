# =====================================
# Codveda Data Analytics Internship
# Level 3 - Task 1
# Classification Analysis
# Dataset: Churn Prediction
# =====================================

import pandas as pd

# Load the datasets
train_df = pd.read_csv("churn-bigml-80.csv")
test_df = pd.read_csv("churn-bigml-20.csv")

# Display the first 5 rows
print(train_df.head())

# Display dataset shape
print(train_df.shape)
# Display dataset information
print(train_df.info())

# Check for missing values
print(train_df.isnull().sum())
from sklearn.preprocessing import LabelEncoder

# Convert categorical columns to numeric
encoder = LabelEncoder()

for column in train_df.select_dtypes(include='object').columns:
    train_df[column] = encoder.fit_transform(train_df[column])
    test_df[column] = encoder.transform(test_df[column])

# Separate features and target
X_train = train_df.drop("Churn", axis=1)
y_train = train_df["Churn"]

X_test = test_df.drop("Churn", axis=1)
y_test = test_df["Churn"]

print("Data preprocessing completed successfully!")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Display Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show(block=False)
plt.pause(5)
plt.close()

print("=" * 40)
print("Codveda Level 3 - Task 1 Completed Successfully!")
print("=" * 40)

input("Press Enter to exit...")