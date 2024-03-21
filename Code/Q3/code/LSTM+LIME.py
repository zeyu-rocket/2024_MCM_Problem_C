from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# 数据预处理
import pandas as pd


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
    epochs=100,
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

# 导入LIME相关的库
from lime import lime_tabular


# 由于LIME需要操作的是原始数据（而不是经过LSTM reshape的数据），我们将创建一个函数来模拟整个预处理加上LSTM模型的预测过程
def lstm_predict(model, data_scaler, X):
    X_scaled = data_scaler.transform(X)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    return model.predict(X_reshaped)


# 创建LIME解释器
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),  # 使用原始训练数据
    feature_names=features,  # 特征名
    mode="regression",
)

# 选择一个实例进行解释
instance_index = 10  # 示例，选择第10个数据点
instance = X_test.iloc[instance_index].values.reshape(1, -1)

# 使用LIME解释该实例的预测
exp = explainer.explain_instance(
    data_row=instance[0], predict_fn=lambda x: lstm_predict(model, scaler, x)
)

# 可视化解释结果
exp.show_in_notebook(show_table=True, show_all=False)

# 如果无法使用 show_in_notebook，可以改用 save_to_file 查看HTML结果
exp.save_to_file(
    "/Users/zeyu/Documents/Learning/数学建模/美赛数学建模/git/comap_mcm_2024/Code/Q3/img/explanation.html"
)


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
