import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")

# Show data
print("Student Data:\n", data)

# Calculate average marks
data["Average"] = data[["Maths", "Science", "English"]].mean(axis=1)

print("\nAverage Marks:\n", data[["Name", "Average"]])

# Find topper
topper = data.loc[data["Average"].idxmax()]
print("\nTopper:\n", topper["Name"])

# Plot graph
data.set_index("Name")[["Maths", "Science", "English"]].plot(kind="bar")

plt.title("Student Marks Comparison")
plt.xlabel("Students")
plt.ylabel("Marks")

plt.show()