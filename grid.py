from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset (update the file path)
file_path = r'C:\Users\Sageer Ansari\Documents\SILKYSKY_DATA_CW2 (S).csv'

df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 1. Encode categorical columns
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Satisfied', 'Age Band', 'Type of Travel', 'Class', 'Destination', 'Continent']

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# 2. Handle missing values for 'Arrival Delay in Minutes'
imputer = SimpleImputer(strategy='mean')
df['Arrival Delay in Minutes'] = imputer.fit_transform(df[['Arrival Delay in Minutes']])

# 3. Split dataset into features (X) and target (y)
X = df.drop(columns=['Satisfied', 'Ref', 'id'])  # Drop the target and irrelevant columns
y = df['Satisfied']

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Define hyperparameters to tune for XGBoost
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],         # Reduced the range for more focused tuning
    'max_depth': [3, 4, 5, 6],                  # Expanded depth options
    'n_estimators': [100, 150, 200],            # Adjusted estimators for better fitting
    'subsample': [0.6, 0.8, 1.0],               # Changed range for subsampling
    'colsample_bytree': [0.6, 0.8, 1.0],        # Changed range for column sampling
    'gamma': [0, 0.1, 0.2, 0.3]                 # Slightly adjusted gamma
}

# 7. Create the GridSearchCV object for XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
grid_search_xgb = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# 8. Fit the grid search to the training data
grid_search_xgb.fit(X_train_scaled, y_train)

# 9. Best model from grid search
best_xgb = grid_search_xgb.best_estimator_

# 10. Predict and evaluate the accuracy of the best model
y_pred_best = best_xgb.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred_best)

# 11. Output results
print("Best Parameters:", grid_search_xgb.best_params_)
print("Best Accuracy:", best_accuracy)
