import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score


# 加载数据
data = pd.read_excel('./lstm_gpt_input.xlsx')  # 请将此路径替换为实际文件路径
data = data.dropna(subset=['y_value_p1', 'y_value_p2'])
y = data['y_value_p1'].values
X = data['elapsed_time_seconds'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
model.fit(X_train_reshaped, y_train, epochs=100, validation_data=(X_test_reshaped, y_test))

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train_reshaped, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping], batch_size=32)

# 使用模型进行预测
predictions = model.predict(X_test_reshaped)

# 反归一化预测
predictions_inverse = scaler.inverse_transform(predictions)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1,1))

# 计算模型评价指标
mse = mean_squared_error(y_test_inverse, predictions_inverse)
r2 = r2_score(y_test_inverse, predictions_inverse)
print(f'MSE: {mse}, R^2: {r2}')

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(y_test_inverse, label='Actual')
plt.plot(predictions_inverse, label='Predicted')
plt.title('LSTM Model Predictions vs Actual')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()