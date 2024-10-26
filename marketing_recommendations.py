
# Step 1: Import necessary libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 2: Fetch the Online Retail dataset
online_retail = fetch_ucirepo(id=352)

# Step 3: Access the features and targets
X = online_retail.data.features
y = online_retail.data.targets

# Step 4: Convert features to a pandas DataFrame for analysis
df = pd.DataFrame(X)

# Step 5: Calculate `TotalSales` as `Quantity * UnitPrice` and use it as the target variable
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# Step 6: Check for NaN values and fill if necessary
df.fillna(0, inplace=True)

# Step 7: Prepare the dataset
X = df[['Quantity', 'UnitPrice']]
y = df['TotalSales']  # Set 'TotalSales' as the target variable

# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 10: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")

# Step 11: Visualization of the predicted vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

