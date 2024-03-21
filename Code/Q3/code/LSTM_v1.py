import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 加载训练集
data_path = "/Users/zeyu/Documents/Learning/数学建模/美赛数学建模/git/comap_mcm_2024/Code/Q3/data/2023-wimbledon-1701_lstm_data.csv"
data = pd.read_csv(data_path)
X = data["score_avg"].values.reshape(-1, 1, 1)

# 划分训练集
train_size = int(len(X) * 0.8)
train = X[0:train_size, :, :]
Y_train = np.roll(train, -1, axis=0)

# 定义LSTM模型
model = Sequential(
    [
        LSTM(units=50, return_sequences=True, input_shape=(1, 1)),
        LSTM(units=50),
        Dense(units=1),
    ]
)
model.compile(optimizer="adam", loss="mean_squared_error")

# 训练模型
model.fit(train, Y_train, epochs=50, batch_size=32)

# 加载验证集
validation_data_path = "/Users/zeyu/Documents/Learning/数学建模/美赛数学建模/git/comap_mcm_2024/Code/Q3/data/2023-wimbledon-1301_lstm_data.csv"
validation_data = pd.read_csv(validation_data_path)
X_val = validation_data["score_avg"].values.reshape(-1, 1, 1)
Y_val = np.roll(X_val, -1, axis=0)  # 假设验证集也是进行一步预测

# 使用验证集进行预测
predictions = model.predict(X_val)

# 将predictions和Y_val从三维数组压缩为一维数组以适应mean_squared_error函数的要求
predictions = predictions.reshape(-1)  # 将形状从[n_samples, 1, 1]压缩为[n_samples]
Y_val = Y_val.reshape(-1)  # 同样，将Y_val压缩为一维数组

# 计算并打印性能指标，忽略最后一个预测值因为没有对应的真实值
mse = mean_squared_error(Y_val[:-1], predictions[:-1])
mae = mean_absolute_error(Y_val[:-1], predictions[:-1])

print(f"Validation Mean Squared Error: {mse}")
print(f"Validation Mean Absolute Error: {mae}")
# 可视化实际值与预测值
plt.figure(figsize=(10, 6))
plt.plot(
    Y_val[:-1], label="Actual", color="blue"
)  # 忽略最后一个值，因为没有对应的预测值
plt.plot(predictions[:-1], label="Predicted", color="red")  # 同样，忽略最后一个预测值
plt.title("Validation Data: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Score Average")
plt.legend()
plt.show()
