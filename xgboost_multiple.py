import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

# Load your dataset (replace with actual path)
data = pd.read_csv('solar_all_params.csv')

# Convert 'period_end' to datetime if not already done
data['period_end'] = pd.to_datetime(data['period_end'])

# Extract time-based features
data['hour'] = data['period_end'].dt.hour
data['day'] = data['period_end'].dt.day
data['month'] = data['period_end'].dt.month
data['day_of_week'] = data['period_end'].dt.dayofweek
data['year'] = data['period_end'].dt.year

# Define features and targets
#features = ['air_temp', 'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 'cloud_opacity',
 #           'precipitation_rate', 'relative_humidity', 'hour', 'day', 'month', 'day_of_week', 'year']
features=['air_temp', 'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi','cloud_opacity','precipitable_water','precipitation_rate', 'relative_humidity','surface_pressure','wind_direction_10m','wind_speed_10m','hour', 'day', 'month', 'day_of_week', 'year']

targets = ['dhi','dni','ghi']

# Feature scaling for inputs
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_features.fit_transform(data[features])

# Dictionary to hold models and scalers for each target
models = {}
scalers_target = {}
y_preds = {}
y_tests = {}
mses = {}
r2_values = {}
# Train a model for each target separately
for target in targets:
    print(f'Training model for {target}')

    # Feature scaling for the target
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(data[[target]])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, scaled_target, test_size=0.2, random_state=42)

    # Reshape the target for training
    y_train = y_train.ravel()  # XGBoost expects a 1D array for the target
    y_test = y_test.ravel()

    # Define the XGBoost model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Inverse transform the predictions to get actual values
    y_pred = scaler_target.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_test = scaler_target.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Store predictions, test values, and model/scaler for each target
    models[target] = model
    scalers_target[target] = scaler_target
    y_preds[target] = y_pred
    y_tests[target] = y_test

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    mses[target] = mse
    print(f'Mean Squared Error for {target}: {mse}\n')

    r2 = r2_score(y_test, y_pred)
    r2_values[target] = r2
    print(f'R² Value for {target}: {r2}\n')

# Plot the predictions vs actual values for each target
plt.figure(figsize=(20, 12))

for idx, target in enumerate(targets):
    plt.subplot(3, 1, idx + 1)
    plt.plot(y_tests[target], label=f'Actual {target.upper()}')
    plt.plot(y_preds[target], label=f'Predicted {target.upper()}')
    plt.title(f'Actual vs Predicted {target.upper()}')
    plt.legend()

plt.tight_layout()
#plt.show()

# Print predicted vs actual values for each target
for target in targets:
    print(f'\nPredicted vs Actual values for {target.upper()}:\n')
    print(f'{"Index":<10} {"Predicted":<15} {"Actual":<15}')
    print('-' * 40)

    for i in range(100):
        print(f'{i:<10} {y_preds[target][i]:<15.2f} {y_tests[target][i]:<15.2f}')









