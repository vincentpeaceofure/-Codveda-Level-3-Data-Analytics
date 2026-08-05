import pandas as pd

# Load the dataset
df = pd.read_csv("3) Sentiment dataset.csv")

# Display the first 5 rows
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display class distribution
print("\nSentiment Distribution:")
print(df.iloc[:, -1].value_counts())

print("=" * 40)
print("Sentiment Dataset Loaded Successfully!")
print("=" * 40)
from textblob import TextBlob

# Create sentiment labels
df['Polarity'] = df.iloc[:, 0].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

print(df[['Polarity']].head())

print("=" * 40)
print("Sentiment Analysis Completed Successfully!")
print("=" * 40)

input("Press Enter to exit...")
import matplotlib.pyplot as plt

# Create sentiment labels
df["Sentiment"] = df["Polarity"].apply(
    lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
)

# Plot sentiment distribution
df["Sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
print("=" * 40)
print("Task 3 Completed Successfully!")
print("=" * 40)

input("Press Enter to exit...")
