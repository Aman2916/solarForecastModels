import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv('solar_data.csv')

# Convert 'period_end' to datetime format
df['period_end'] = pd.to_datetime(df['period_end'])

# Extract time-based features from 'period_end'
df['hour'] = df['period_end'].dt.hour
df['day'] = df['period_end'].dt.day
df['month'] = df['period_end'].dt.month
df['day_of_week'] = df['period_end'].dt.dayofweek  # Monday=0, Sunday=6
df['year']=df['period_end'].dt.year

# Create lag features
df['solar_radiation_lag1'] = df['ghi'].shift(1)  # Lag by 1 hour
df['solar_radiation_lag2'] = df['ghi'].shift(2)  # Lag by 2 hours

# Add more lags if necessary

# Drop rows with NaN values resulting from lagging
df = df.dropna()

# Drop the original 'period_end' and 'period' columns if not needed
df = df.drop(columns=['period', 'period_end'])

# Define features and target
X = df[['air_temp','clearsky_dhi','clearsky_dni','clearsky_ghi','cloud_opacity','precipitation_rate','relative_humidity','hour', 'day', 'month', 'day_of_week',
     'year']]
     # Define target variables (Y)

Y = df[['dni', 'ghi', 'dhi']]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = model.predict(X_test)


# Evaluate the model
mse_dni = mean_squared_error(Y_test['dni'], Y_pred[:, 0])
mse_ghi = mean_squared_error(Y_test['ghi'], Y_pred[:, 1])
mse_dhi = mean_squared_error(Y_test['dhi'], Y_pred[:, 2])

# Print results
print(f'Mean Squared Error for DNI: {mse_dni}')
print(f'Mean Squared Error for GHI: {mse_ghi}')
print(f'Mean Squared Error for DHI: {mse_dhi}')

# Optional: To check model coefficients
print(f"Model coefficients: {model.coef_}")
print(f"Model intercept: {model.intercept_}")


new_data = pd.DataFrame({

    'air_temp': [15],
   'clearsky_dhi':[237],
   'clearsky_dni':[220],
   'clearsky_ghi':[352],
   'cloud_opacity':[18.9],
    'precipitation_rate':[0],
    'relative_humidity':[84.7],
    'hour': [5],
    'day': [1],
    'month': [1],
    'day_of_week':1,
    'year':[2020],
     })

predicted_ghi = model.predict(new_data)

# Output the predicted GHI
print(f'Predicted DNI GHI DHI: {predicted_ghi[0]}')







"""
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Show the actual vs predicted values
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print("\nActual vs Predicted:")
print(results)

#import matplotlib.pyplot as plt

# Define the number of records to display (e.g., first 100 records)
num_records = 10

# Select the first 'num_records' from y_test and y_pred
y_test_subset = y_test[:num_records]
y_pred_subset = y_pred[:num_records]
time_feature_subset = df.loc[y_test.index[:num_records], 'year']  # Use the appropriate time feature like 'hour'

# Plot the subset of actual vs predicted values
#plt.figure(figsize=(10, 6))

# Plot actual values for the first 'num_records'
#plt.plot(time_feature_subset, y_test_subset.values, label='Actual Values', color='blue', marker='o')

# Plot predicted values for the first 'num_records'
#plt.plot(time_feature_subset, y_pred_subset, label='Predicted Values', color='red', linestyle='dashed', marker='x')

# Add titles and labels
#plt.title(f'Actual vs Predicted Values (First {num_records} Records)')
#plt.xlabel('Year')
#plt.ylabel('Target Variable (e.g., Solar Radiation)')

# Show the legend
#plt.legend()

# Show the plot
#plt.show()
# Plot residuals to check for any patterns
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, color='green', marker='o')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Actual GHI')
plt.ylabel('Residuals')
plt.show()

new_data = pd.DataFrame({

     'air_temp': [25],
   'clearsky_dhi':[164],
   'clearsky_dni':[773],
   'clearsky_ghi':[927],
   'cloud_opacity':[86],
    'precipitation_rate':[0.7],
    'relative_humidity':[96.5],
    'hour': [7],
    'day': [26],
    'month': [7],
    'day_of_week':[5],
    'year':[2024],
    'solar_radiation_lag1':[130],
    'solar_radiation_lag2':[100]
     })
#25	164	773	927	86	130	0	130	0.7	96.5	2024-07-26T07:00:00+00:00


# Use the trained model to predict GHI
predicted_ghi = model.predict(new_data)

# Output the predicted GHI
print(f'Predicted GHI: {predicted_ghi[0]}')

"""



