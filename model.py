import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

csv_path = 'solar_data.csv'
df = pd.read_csv(csv_path)

# Preprocess the data
df['period_end'] = pd.to_datetime(df['period_end'])
df['hour'] = df['period_end'].dt.hour
df['day'] = df['period_end'].dt.day
df['month'] = df['period_end'].dt.month
df['day_of_week'] = df['period_end'].dt.dayofweek
df['year'] = df['period_end'].dt.year

X = df[['air_temp', 'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 'cloud_opacity', 'precipitation_rate', 'relative_humidity', 'hour', 'day', 'month', 'day_of_week', 'year']]
Y = df[['dni', 'ghi', 'dhi']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Save the trained model to a file
joblib.dump(model, 'solar_model.pkl')
