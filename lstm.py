import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
# Load your dataset (replace with the actual path)
data = pd.read_csv('solar_data.csv')
data['period_end'] = pd.to_datetime(data['period_end'])

# Extract time-based features from 'period_end'
data['period_end'] = pd.to_datetime(data['period_end'])  # Convert period_end to datetime if it's not already

# Extract components from the period_end column
data['hour'] = data['period_end'].dt.hour
data['day'] = data['period_end'].dt.day
data['month'] = data['period_end'].dt.month
data['day_of_week'] = data['period_end'].dt.dayofweek  # Monday=0, Sunday=6
data['year'] = data['period_end'].dt.year

# Let's assume the columns are: 'temp', 'humidity', 'clear_sky_dni', 'dhi', 'ghi', 'day', 'hour', 'month', 'year'
# Extract relevant features and the target variable (GHI)
features=['air_temp','clearsky_dhi','clearsky_dni','clearsky_ghi','dhi','dni','cloud_opacity','precipitation_rate','relative_humidity','hour', 'day', 'month', 'day_of_week',
     'year']
target = 'ghi'

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[features])
scaled_target = scaler.fit_transform(data[[target]])

# Combine the scaled features and target into a new dataframe
scaled_data = pd.DataFrame(scaled_features, columns=features)
scaled_data[target] = scaled_target

# Prepare the dataset for LSTM
def create_dataset(dataset, target, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step, :-1])  # all features except the target
        y.append(dataset[i + time_step, -1])  # target column (GHI)
    return np.array(X), np.array(y)

# Define time step (how many previous time steps to look at)
time_step = 60  # Using past 60 time steps (hours) to predict the next hour's GHI

# Create the dataset for LSTM
X, y = create_dataset(np.hstack((scaled_features, scaled_target)), target, time_step)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape input to be [samples, time steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(features))
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], len(features))

# Build the LSTM model

model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)


# Predict and inverse transform the predictions
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(np.concatenate((np.zeros((predicted.shape[0], len(features))), predicted), axis=1))[:, -1]
y_test = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], len(features))), y_test.reshape(-1, 1)), axis=1))[:, -1]


# Assuming y_test contains true target values
predictions_df = pd.DataFrame({
    'Actual GHI': y_test.flatten(),
    'LSTM Predicted GHI': predicted.flatten()
})

# Save to a CSV for further analysis or use
predictions_df.to_csv('lstm_predictions.csv', index=False)

# Evaluate the model
mse = np.mean((y_test - predicted) ** 2)
print(f'Mean Squared Error: {mse}')

print("\nActual vs Predicted for GHI:")
for actual, predicted in zip(y_test['ghi'], predicted[:, 1]):
    print(f"Actual: {actual}, Predicted: {predicted}")

# Plot the predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual GHI')
plt.plot(predicted, label='Predicted GHI')
plt.legend()
#plt.show()
