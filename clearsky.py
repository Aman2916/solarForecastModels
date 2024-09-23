import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset

data = pd.read_csv('solar_all_params.csv')

data['period_end'] = pd.to_datetime(data['period_end'])

# Extract time-based features
data['hour'] = data['period_end'].dt.hour
data['day'] = data['period_end'].dt.day
data['month'] = data['period_end'].dt.month
data['day_of_week'] = data['period_end'].dt.dayofweek
data['year'] = data['period_end'].dt.year

# Define features and target
#features = ['air_temp','azimuth','cloud_opacity',
 #           'precipitation_rate', 'relative_humidity','surface_pressure','wind_speed_10m','zenith', 'hour', 'day', 'month', 'day_of_week', 'year']

features = ['air_temp','albedo','azimuth','clearsky_dhi','clearsky_ghi','relative_humidity','surface_pressure','wind_speed_10m','precipitable_water','zenith', 'hour', 'day', 'month', 'day_of_week', 'year']
# air_temp				clearsky_dni		clearsky_gti	cloud_opacity	dhi	dni	ghi	gti		precipitation_rate	relative_humidity		wind_direction_100m	wind_direction_10m	wind_speed_100m		period_end	period]
target = 'clearsky_dni'

# Feature scaling using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[features])
scaled_target = scaler.fit_transform(data[[target]])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, scaled_target, test_size=0.2, random_state=42)

# Reshape the target for training
y_train = y_train.ravel()  # XGBoost expects a 1D array for the target
y_test = y_test.ravel()

# Define the XGBoost model
model =xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Train the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions to get actual DNI values
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()


from sklearn.metrics import mean_squared_error

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual DNI')
plt.plot(y_pred, label='Predicted DNI')
plt.legend()
#plt.show()

# Get the feature importances
importances = model.feature_importances_

# Print the feature importances alongside feature names
for feature, importance in zip(scaled_features, importances):
    print(f"Feature: {feature}, Importance: {importance}")
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Plot the feature importances
plt.figure(figsize=(10, 8))
plot_importance(model, importance_type='weight')  # Choose 'weight', 'gain', or 'cover'
plt.show()



