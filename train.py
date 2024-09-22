import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Features (X) and target (y)
X = df[['temperature', 'humidity', 'cloud_cover']]
y = df['solar_radiation']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model (make predictions on the test set)
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Display predictions alongside actual values
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print("\nActual vs Predicted:")
print(results)
