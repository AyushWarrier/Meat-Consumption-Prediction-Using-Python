import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

file_path = "/home/ayushwarrier/Documents/Machine Learning Project/meat_consumption.csv" 
df = pd.read_csv(file_path)

print("Dataset Overview:")
print(df.info())  # Shows column names, data types, and missing values

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:") # Checking for any missing values
print(df.isnull().sum())

# Set plot style
sns.set_style("whitegrid")

plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["Total Meat Consumption"], marker='o', linestyle='-', color='b', label="Total Meat Consumption")
plt.xlabel("Year")
plt.ylabel("Meat Consumption per Capita")
plt.title("Meat Consumption Trend in India (1961-2021)")
plt.legend()
plt.show()

X = df["Year"].values.reshape(-1, 1)  # Independent variable (Year)
y = df["Total Meat Consumption"].values  # Dependent variable (Consumption)

model = LinearRegression()
model.fit(X, y)

# Predict Future Consumption (For the next 10 Years i.e. 2022 - 2032)
future_years = np.arange(df["Year"].max() + 1, df["Year"].max() + 11).reshape(-1, 1)
future_predictions = model.predict(future_years)

# Visualize
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='green', linestyle="--", label="Trend Line (Model Fit)")
plt.plot(future_years, future_predictions, color='red', marker='o', label="Future Predictions")
plt.xlabel("Year")
plt.ylabel("Total Meat Consumption")
plt.title("Meat Consumption Prediction in India (2022-2032)")
plt.legend()
plt.show()

y_pred = model.predict(X)

# Calculate the evaluation metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Print the results
print(f"R² Score: {r2:.4f}")  # R² should be close to 1 for a good model
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

metrics = ["R² Score", "MAE", "RMSE"]
values = [0.9985, 0.55, 0.75]

plt.figure(figsize=(6,4))
plt.bar(metrics, values, color=["green", "orange", "red"])
plt.title("Comparison of Regression Metrics")
plt.ylabel("Metric Value")
plt.ylim(0, 1.2) 
plt.show()
