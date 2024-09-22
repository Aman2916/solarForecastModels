import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load your CSV file
df = pd.read_csv('solar_all_params.csv')

# Convert 'period_end' to datetime format
df['period_end'] = pd.to_datetime(df['period_end'])

# Extract time-based features from 'period_end'
df['hour'] = df['period_end'].dt.hour
df['day'] = df['period_end'].dt.day
df['month'] = df['period_end'].dt.month
df['day_of_week'] = df['period_end'].dt.dayofweek  # Monday=0, Sunday=6
df['year'] = df['period_end'].dt.year

# Define input features (X)
X = df[['air_temp', 'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi','clearsky_gti','cloud_opacity','precipitable_water','precipitation_rate', 'relative_humidity','surface_pressure','wind_direction_10m','wind_speed_10m','hour', 'day', 'month', 'day_of_week', 'year']]


# Define target variables (Y)
Y = df[['dhi','dni','ghi','gti']]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor for multivariate output
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = model.predict(X_test)

# Evaluate the model (for each target)
mse_dhi = mean_squared_error(Y_test['dhi'], Y_pred[:, 0])
mse_dni = mean_squared_error(Y_test['dni'], Y_pred[:, 1])
mse_ghi = mean_squared_error(Y_test['ghi'], Y_pred[:, 2])
mse_gti = mean_squared_error(Y_test['gti'], Y_pred[:, 3])

print(f'MSE for DNI: {mse_dni}')
print(f'MSE for GTI: {mse_gti}')
print(f'MSE for GHI: {mse_ghi}')
print(f'MSE for DHI: {mse_dhi}\n')

# Print actual vs predicted values for each target
print("Actual vs Predicted for DNI:")
for actual, predicted in zip(Y_test['dhi'], Y_pred[:, 0]):
    print(f"Actual: {actual}, Predicted: {predicted}")

print("\nActual vs Predicted for DHI:")
for actual, predicted in zip(Y_test['dni'], Y_pred[:, 1]):
    print(f"Actual: {actual}, Predicted: {predicted}")

print("\nActual vs Predicted for GHI:")
for actual, predicted in zip(Y_test['ghi'], Y_pred[:, 2]):
    print(f"Actual: {actual}, Predicted: {predicted}")

print("\nActual vs Predicted for GTI:")
for actual, predicted in zip(Y_test['gti'], Y_pred[:, 3]):
    print(f"Actual: {actual}, Predicted: {predicted}")


