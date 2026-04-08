import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("sales.csv")

print("Sales Data:\n", data)

# Total sales
total_sales = data["Sales"].sum()
print("\nTotal Sales:", total_sales)

# Best selling product
best_product = data.groupby("Product")["Sales"].sum().idxmax()
print("\nBest Selling Product:", best_product)

# Highest sales day
best_day = data.loc[data["Sales"].idxmax()]
print("\nHighest Sales Day:", best_day["Date"])

# Plot graph
data.plot(x="Date", y="Sales", kind="line")

plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.show()