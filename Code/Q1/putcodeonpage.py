import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data_path = "../data/动量计算_0_1_raw.csv"
data = pd.read_csv(data_path)

# 数据预处理
data["p1_server"] = data["p1_server"].replace(2, 0)
data["p1_server"] = data["p1_server"].replace(1, 0.25)
data["p2_server"] = data["p2_server"].replace(1, 0)
data["p2_server"] = data["p2_server"].replace(2, 0.25)
data["p2_Players_run_distance_difference"] = (-1) * data[
    "p2_Players_run_distance_difference"
]
data["p2_Player_socre_margin"] = (-1) * data["p2_Player_socre_margin"]

# 修正发球成功率和得分率，将0值替换为1
data["p1_1st_serve_success_ratio"] = data["p1_1st_serve_success_ratio"].replace(0, 1)
data["p2_1st_serve_success_ratio"] = data["p1_1st_serve_success_ratio"].replace(0, 1)
data["p1_1st_serve_win_ratio"] = data["p1_1st_serve_win_ratio"].replace(0, 1)
data["p2_1st_serve_win_ratio"] = data["p2_1st_serve_win_ratio"].replace(0, 1)
data["p1_2nd_serve_win_ratio"] = data["p1_2nd_serve_win_ratio"].replace(0, 1)
data["p1_2nd_serve_win_ratio"] = data["p1_2nd_serve_win_ratio"].replace(0, 1)
data["p2_2nd_serve_win_ratio"] = data["p2_2nd_serve_win_ratio"].replace(0, 1)


# 归一化处理
def normalize_column_to_0_1(column):
    return (column - column.min()) / (column.max() - column.min())


data["serve_no"] = normalize_column_to_0_1(data["serve_no"])
data["serve_width"] = normalize_column_to_0_1(data["serve_width"])
data["serve_depth"] = normalize_column_to_0_1(data["serve_depth"])
data["continuous_wins_p1"] = normalize_column_to_0_1(data["continuous_wins_p1"])
data["continuous_wins_p2"] = normalize_column_to_0_1(data["continuous_wins_p2"])
data["p1_ace_count2r"] = normalize_column_to_0_1(data["p1_ace_count2r"])
data["p2_ace_count2r"] = normalize_column_to_0_1(data["p1_ace_count2r"])
data["p1_unf_err_count2r"] = normalize_column_to_0_1(data["p1_ace_count2r"]) * (-1)
data["p2_unf_err_count2r"] = normalize_column_to_0_1(data["p1_ace_count2r"]) * (-1)


# 更多的归一化和数据预处理
def normalize_to_range(column, new_min=-1, new_max=1):
    return ((column - column.min()) / (column.max() - column.min())) * (
        new_max - new_min
    ) + new_min


data["p1_Player_socre_margin"] = normalize_to_range(data["p1_Player_socre_margin"])
data["p2_Player_socre_margin"] = normalize_to_range(data["p2_Player_socre_margin"])
data["speed_mph"] = normalize_to_range(data["speed_mph"])
data["Players_run_distance_difference"] = normalize_to_range(data["speed_mph"])
data["p1_Players_run_distance_difference"] = normalize_to_range(
    data["p1_Players_run_distance_difference"]
)
data["p2_Players_run_distance_difference"] = normalize_to_range(
    data["p2_Players_run_distance_difference"]
)

# 计算并可视化动量值
features_weights = {
    # 特征权重映射，此处省略具体数值填充
}


def calculate_momentum(data, features_weights):
    # 根据特征权重计算动量值
    pass  # 实现动量值计算逻辑


# 使用matplotlib绘制动量图
def plot_momentum(data):
    plt.figure(figsize=(10, 6))
    # 绘图逻辑，此处省略
    plt.show()
