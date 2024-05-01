import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('AESO_2020_demand_price.csv')

print(data.head())

# Extract relevant features
xData_raw = data[['Price ($)', '30Ravg ($)', 'AIL Demand (MW)']].values
yData_raw = data['AIL Demand (MW)'].values.reshape(-1, 1)

# Feature scaling
scaler_x = MinMaxScaler(feature_range=(0, 1))
xData_scaled = scaler_x.fit_transform(xData_raw)

scaler_y = MinMaxScaler(feature_range=(0, 1))
yData_scaled = scaler_y.fit_transform(yData_raw)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(xData_scaled, yData_scaled, test_size=0.2, shuffle=False)

# Reshape input data for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

# Training Phase
model.summary()

# Predict on test data
y_predict = model.predict(X_test)

# Inverse transform predictions
y_predict_actual = scaler_y.inverse_transform(y_predict)
y_test_actual = scaler_y.inverse_transform(y_test)

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test_actual - y_predict_actual) / y_test_actual)) * 100
print("Mean Absolute Percentage Error (MAPE):", mape)

# Plot predictions vs actual
plt.plot(y_predict_actual, label='Predicted Demand')
plt.plot(y_test_actual, label='Actual Demand')
plt.xlabel('Time')
plt.ylabel('AIL Demand (MW)')
plt.title('Demand Prediction using LSTM')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()