import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

# 加载数据
data_path = "../Q1/动量计算_完成_v1.csv"  # 请替换为您的数据文件路径
data = pd.read_csv(data_path)

# 分别对两个指标进行变点检测
series_names = ['y_value_p2', 'y_value_p1']
results = {}  # 用于存储两个指标的变点检测结果

for series_name in series_names:
    data_series = data[series_name].values
    model = "l2"
    algo = rpt.Pelt(model=model).fit(data_series)
    result = algo.predict(pen=10)
    results[series_name] = result

# 可视化
plt.figure(figsize=(12, 6))
colors = ['blue', 'green']  # 为两个指标分别指定颜色
for i, series_name in enumerate(series_names):
    data_series = data[series_name].values
    plt.plot(data_series, label=series_name, color=colors[i])
    for pt in results[series_name][:-1]:  # 绘制变点
        plt.axvline(x=pt, color=colors[i], linestyle='--', lw=1)
        plt.plot(pt, data_series[pt], 'o', color=colors[i])

plt.title('Change Point Detection for Multiple Indicators')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# 初始化一个空的DataFrame，用于存储所有变点的数据
change_points_all = pd.DataFrame(columns=['Indicator', 'Index', 'Value'])

# 遍历每个指标及其变点检测结果
for i, series_name in enumerate(series_names):
    data_series = data[series_name].values
    change_points = [(series_name, index, data_series[index]) for index in results[series_name][:-1]]  # 获取变点的索引和值
    change_points_df = pd.DataFrame(change_points, columns=['Indicator', 'Index', 'Value'])
    change_points_all = pd.concat([change_points_all, change_points_df], ignore_index=True)

# 保存到CSV文件
change_points_all.to_csv("change_points_multiple_indicators_v2.csv", index=False)
