import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("flood.csv")

# Preview
print(data.head())

# Features (inputs)
X = data.drop("FloodProbability", axis=1) #removed target value column

# Target (output)
y = data["FloodProbability"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print("\nModel Accuracy:", score)

# Prediction (sample input)
sample = X.iloc[0:1]
prediction = model.predict(sample)

print("\nPredicted Flood Probability:", prediction[0])


import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Flood Prediction Accuracy")
plt.show()