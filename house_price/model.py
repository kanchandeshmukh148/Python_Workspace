import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("data.csv")

# Input (X) and Output (y)
X = data[["Size"]]
y = data["Price"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price
size = [[2200]]
prediction = model.predict(size)

print("Predicted Price:", prediction[0])