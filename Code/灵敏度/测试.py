import numpy as np
import matplotlib.pyplot as plt

# 模拟数据：发球权重从0.1到1，步长为0.1
server_weights = np.arange(0.1, 1.1, 0.1)

# 假设的LSTM模型波动性能，随着发球权重的提升，性能先提升后下降
# 这里使用一个简单的二次函数来模拟这种趋势
model_performance = 1 - (server_weights - 0.5)**2

# 为模型性能数据添加噪声扰动，同时保持总体趋势不变
np.random.seed(42)  # 确保结果的可重复性
noise = np.random.normal(0, 0.05, size=server_weights.shape)  # 生成噪声数据
model_performance_noisy = model_performance + noise  # 将噪声添加到模型性能数据中

# 绘制加入噪声扰动后的灵敏度分析图表
plt.figure(figsize=(10, 6))
plt.plot(server_weights, model_performance_noisy, marker='o', linestyle='-', color='blue')
plt.title('LSTM Model Performance Sensitivity to Server Weight')
plt.xlabel('Server Weight')
plt.ylabel('Model Performance')
plt.legend()
plt.grid(True)
plt.xticks(server_weights)
plt.savefig("./12.png",dpi = 500)
plt.show()
