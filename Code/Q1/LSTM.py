from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import numpy as np

# 假设X_train是我们的输入特征，y_train是标签（0或1表示输赢）
# 以下代码假设X_train, y_train已经准备好并且是适当的格式

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_reshaped = X_train_scaled.reshape((X_train.shape[0], 1, X_train.shape[1])) # LSTM需要的格式 [样本数, 时间步长, 特征数]


model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[1]))) # 根据需要调整LSTM单元数和激活函数
model.add(Dense(1, activation='sigmoid')) # 输出层，使用sigmoid激活函数来预测二分类问题

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_reshaped, y_train, epochs=100, validation_split=0.2) # 根据需要调整训练轮次和验证集大小

# 假设X_test, y_test已经准备好并且是适当的格式，并且已经进行了相同的预处理步骤
X_test_scaled = scaler.transform(X_test)
X_test_reshaped = X_test_scaled.reshape((X_test.shape[0], 1, X_test.shape[1]))
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f'Test accuracy: {accuracy}')
