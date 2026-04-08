import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("flood.csv")
print(data.head())

print(data.info())
print(data.describe())

print(data.isnull().sum())

data = data.dropna()

data = data.drop_duplicates()

X = data.drop("FloodProbability", axis=1)
y = data["FloodProbability"]

data.hist(figsize=(10,8))
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Flood Probability")
plt.show()

sample = X.iloc[1:4]
prediction = model.predict(sample)

print("Predicted Flood Probability:", prediction[0])