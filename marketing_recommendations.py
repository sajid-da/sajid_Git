# Step 1: Import necessary libraries
from ucimlrepo import fetch_ucirepo  # Ensure correct library or replace with suitable data fetching method
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 2: Fetch the Online Retail dataset
online_retail = fetch_ucirepo(id=352)

# Step 3: Access the features and targets
X = online_retail.data.features
y = online_retail.data.targets

# Step 4: Convert features to a pandas DataFrame for analysis
df = pd.DataFrame(X)
df['Target'] = y  # Adding the target variable to the DataFrame

# Step 5: Check for NaN values in the DataFrame
print("Missing values before handling:")
print(df.isna().sum())

# Fill or drop NaN values (Example: Filling with 0)
df.fillna(0, inplace=True)

# Verify missing values have been handled
print("Missing values after handling:")
print(df.isna().sum())

# Step 6: Prepare the dataset for training
X = df[['Quantity', 'UnitPrice']]  # Modify based on relevant features in your dataset
y = df['Target']  # Target variable

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)

# Check if there are NaN values in the predictions
if np.isnan(y_pred).any():
    print("Warning: NaN values detected in predictions.")
else:
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

# Step 10: Create a function for user input and prediction
def predict_sales(quantity, unit_price):
    # Handle potential NaN inputs in user data
    if np.isnan(quantity) or np.isnan(unit_price):
        print("Invalid input: Quantity and Unit Price cannot be NaN.")
        return np.nan
    
    # Create a DataFrame from user input
    user_input = pd.DataFrame([[quantity, unit_price]], columns=['Quantity', 'UnitPrice'])
    
    # Predict using the trained model
    predicted_sales = model.predict(user_input)
    return predicted_sales[0]  # Return the prediction

# Step 11: Example of user input
if __name__ == "__main__":
    try:
        quantity = float(input("Enter quantity: "))
        unit_price = float(input("Enter unit price: "))
        
        # Make prediction
        prediction = predict_sales(quantity, unit_price)
        if np.isnan(prediction):
            print("Prediction could not be made due to invalid inputs.")
        else:
            print(f"Predicted Sales: {prediction:.2f}")
    except ValueError:
        print("Please enter valid numeric inputs for quantity and unit price.")
