# 数据预处理
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 加载数据
train_data = pd.read_csv(
    "/Users/zeyu/Documents/Learning/数学建模/美赛数学建模/git/comap_mcm_2024/Code/Q3/data/momentum_v1_train.csv"
)
test_data = pd.read_csv(
    "/Users/zeyu/Documents/Learning/数学建模/美赛数学建模/git/comap_mcm_2024/Code/Q3/data/momentum_v1_test.csv"
)

# 选择特征和标签
features = [
    "p1_1st_serve_success_ratio",
    "p1_1st_serve_win_ratio",
    "p1_2nd_serve_win_ratio",
    "p1_Players_run_distance_difference",
    "p1_server",
    "p1_Player_socre_margin",
]
X_train = train_data[features]
y_train = train_data["y_value_p1_normalized_0_5"]
X_test = test_data[features]
y_test = test_data["y_value_p1_normalized_0_5"]


# 移除数据中的空值（NAN）
train_data["y_value_p1_normalized_0_5"].fillna(
    train_data["y_value_p1_normalized_0_5"].mean(), inplace=True
)
test_data["y_value_p1_normalized_0_5"].fillna(
    test_data["y_value_p1_normalized_0_5"].mean(), inplace=True
)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 重塑数据以适应LSTM模型
X_train_reshaped = X_train_scaled.reshape(
    (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
)
X_test_reshaped = X_test_scaled.reshape(
    (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
)
# LSTM 模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 构建模型
model = Sequential(
    [LSTM(50, activation="relu", input_shape=(1, X_train_scaled.shape[1])), Dense(1)]
)
model.compile(optimizer="adam", loss="mse")

# 训练模型
history = model.fit(
    X_train_reshaped,
    y_train,
    epochs=10,
    validation_data=(X_test_reshaped, y_test),
    batch_size=32,
    verbose=1,
)


# 可视化训练过程
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="test")
plt.title("LSTM Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 输出损失
test_loss = model.evaluate(X_test_reshaped, y_test, verbose=1)
print(f"Test Loss: {test_loss}")


test_loss = model.evaluate(X_test_reshaped, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
# 进行预测
y_pred = model.predict(X_test_reshaped)

# 可视化预测结果与实际值
plt.figure(figsize=(10, 6))
plt.plot(y_test.reset_index(drop=True), label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("LSTM Model Predictions vs Actual")
plt.xlabel("Sample")
plt.ylabel("Value")
plt.legend()
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
