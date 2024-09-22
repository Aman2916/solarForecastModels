import pandas as pd

# Assuming you have XGBoost predictions and LSTM predictions
# 'xgboost_predictions' and 'lstm_predictions' contain predictions, and 'y_test' is the true target

# Create a DataFrame to store actual values and predictions from both models
predictions_df = pd.DataFrame({
    'Actual GHI': y_test,
    'XGBoost Predicted GHI': xgboost_predictions,
    'LSTM Predicted GHI': lstm_predictions
})

# Save to a CSV file
predictions_df.to_csv('xgboost_lstm_predictions.csv', index=False)

print("Predictions from XGBoost and LSTM saved to 'xgboost_lstm_predictions.csv'.")
