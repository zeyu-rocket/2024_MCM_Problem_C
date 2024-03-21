import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# 可视化
import matplotlib.pyplot as plt

# 加载数据
train_data = pd.read_csv(
    "/Users/zeyu/Documents/Learning/数学建模/美赛数学建模/git/comap_mcm_2024/Code/Q3/data/filtered_data_train.csv"
)

test_data = pd.read_csv(
    "/Users/zeyu/Documents/Learning/数学建模/美赛数学建模/git/comap_mcm_2024/Code/Q3/data/filtered_data_test_min.csv"
)


# 选择特征和目标
feature_columns = train_data.columns.drop(
    "value_1"
)  # 假设除了point_victor外的所有列都是特征
target_column = "value_1"

# 移除数据中的空值（NAN）
train_data["value_1"].fillna(train_data["value_1"].mean(), inplace=True)
test_data["value_1"].fillna(test_data["value_1"].mean(), inplace=True)

X_train_raw = train_data[feature_columns]
y_train_raw = train_data[target_column]
X_test_raw = test_data[feature_columns]
y_test_raw = test_data[target_column]

# 归一化数据
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


# 重构数据为时间序列
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i : (i + time_steps)]
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 1
X_train, y_train = create_dataset(X_train_scaled, y_train_raw, time_steps)
X_test, y_test = create_dataset(X_test_scaled, y_test_raw, time_steps)

print("X_train_raw中的NaN值数量:", X_train_raw.isnull().any().sum())
print("y_train_raw中的NaN值数量:", y_train_raw.isnull().sum())
print("X_test_raw中的NaN值数量:", X_test_raw.isnull().any().sum())
print("y_test_raw中的NaN值数量:", y_test_raw.isnull().sum())

# 定义LSTM模型
model = Sequential(
    [LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])), Dense(1)]
)

model.compile(optimizer="adam", loss="mean_squared_error")

# 训练模型，并保存历史信息用于可视化
history = model.fit(
    X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
)

# 评估模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")

# 阈值化预测值
y_pred_thresholded = (y_pred > 0.5).astype(int)

# 可视化
# 可视化训练和验证损失
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 可视化阈值化后的预测值和实际值
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual", linestyle="-", marker="o")
plt.plot(
    y_pred_thresholded.flatten(),
    label="Predicted (Thresholded)",
    linestyle="-",
    marker="x",
)
plt.title("Actual vs. Predicted (Thresholded) Values")
plt.xlabel("Sample")
plt.ylabel("Point Victor")
plt.legend()
plt.show()
