# training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from sklearn.preprocessing import LabelEncoder


# Load the Data from JSON Files
print("Loading data from JSON files...")
train_df = pd.read_json('train_df.json', orient='records', lines=True)  # Adjust the path as needed


label_encoder = LabelEncoder()
train_df['text_encoded'] = label_encoder.fit_transform(train_df['text'])
# Features and Target
X = train_df[['quarter', 'year', 'text_encoded']]  # Update with your feature columns
y = train_df['value_standardized']  # Update with your target column

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Regressor with Best Parameters
print("Training Random Forest Regressor with Best Parameters...")
rf_model_best = RandomForestRegressor(
    max_depth=20,
    min_samples_split=10,
    n_estimators=500,
    random_state=42
)
rf_model_best.fit(X_train_scaled, y_train)

# Make Predictions
y_train_pred_best = rf_model_best.predict(X_train_scaled)
y_test_pred_best = rf_model_best.predict(X_test_scaled)

# Evaluate the Model
mse_train_best = mean_squared_error(y_train, y_train_pred_best)
r2_train_best = r2_score(y_train, y_train_pred_best)
mae_train_best = mean_absolute_error(y_train, y_train_pred_best)

mse_test_best = mean_squared_error(y_test, y_test_pred_best)
r2_test_best = r2_score(y_test, y_test_pred_best)
mae_test_best = mean_absolute_error(y_test, y_test_pred_best)

# Print Results
print(f"Random Forest with Best Parameters - Training Metrics:")
print(f"  MSE: {mse_train_best:.4f}")
print(f"  R²: {r2_train_best:.4f}")
print(f"  MAE: {mae_train_best:.4f}")

print(f"Random Forest with Best Parameters - Testing Metrics:")
print(f"  MSE: {mse_test_best:.4f}")
print(f"  R²: {r2_test_best:.4f}")
print(f"  MAE: {mae_test_best:.4f}")

# Save the Model and Scaler
print("Saving the model and scaler...")
with open('rf_model_best.pkl', 'wb') as model_file:
    pickle.dump(rf_model_best, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully.")
