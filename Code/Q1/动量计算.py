import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_path = "../data/动量计算_0_1_raw.csv"

data = pd.read_csv(data_path)

data['p1_server'] = data['p1_server'].replace(2, 0)
data['p1_server'] = data['p1_server'].replace(1, 0.25)
data['p2_server'] = data['p2_server'].replace(1, 0)
data['p2_server'] = data['p2_server'].replace(2, 0.25)
data["p2_Players_run_distance_difference"] = (-1) * data["p2_Players_run_distance_difference"]
data["p2_Player_socre_margin"] = (-1) * data["p2_Player_socre_margin"]

data["p1_1st_serve_success_ratio"] = data["p1_1st_serve_success_ratio"].replace(0, 1)
data["p2_1st_serve_success_ratio"] = data["p1_1st_serve_success_ratio"].replace(0, 1)
data["p1_1st_serve_win_ratio"] = data["p1_1st_serve_win_ratio"].replace(0, 1)
data["p2_1st_serve_win_ratio"] = data["p2_1st_serve_win_ratio"].replace(0, 1)
data["p1_2nd_serve_win_ratio"] = data["p1_2nd_serve_win_ratio"].replace(0, 1)
data["p1_2nd_serve_win_ratio"] = data["p1_2nd_serve_win_ratio"].replace(0, 1)
data["p2_2nd_serve_win_ratio"] = data["p2_2nd_serve_win_ratio"].replace(0, 1)

# %%
features_weights = {
    "p1 Players run distance difference": 7.333,
    "p2 Players run distance difference": 7.333,
    "p1 unf err count2r": 6.4,
    "speed mph": 5.9,
    "p2 unf err count2r": 6.2,
    "p1 2nd serve win ratio": 4.78,
    "p2 1st serve win ratio": 5.93,
    "p1 1st serve win ratio": 6.25,
    "p2 2nd serve win ratio": 5.25,
    "p2 1st serve success ratio": 6.1,
    "p1 1st serve success ratio": 5.5833,
    "p1 Player socre margin": 8.6166,
    "p2 Player socre margin": 8.6166,
    "continuous wins p1": 4,
    "continuous wins p2": 4.21667,
    "serve width": 2.15,
    "serve no": 2.25,
    "p2 ace count2r": 1.9833,
    "serve depth": 1.1166,  # Assuming last serve depth is a different measure or corrected value
    "p1 ace count2r": 1.4166,
    "p1 server": 18.68333,
    "p2 server": 18.68333
}


def normalize_column_to_0_1(column):
    return (column - column.min()) / (column.max() - column.min()) * 1

data["serve_no"] = normalize_column_to_0_1(data["serve_no"])
data["serve_width"] = normalize_column_to_0_1(data["serve_width"])
data["serve_depth"] = normalize_column_to_0_1(data["serve_depth"])
data["continuous_wins_p1"] = normalize_column_to_0_1(data["continuous_wins_p1"])
data["continuous_wins_p2"] = normalize_column_to_0_1(data["continuous_wins_p2"])

data["p1_ace_count2r"] = normalize_column_to_0_1(data["p1_ace_count2r"])
data["p2_ace_count2r"] = normalize_column_to_0_1(data["p1_ace_count2r"])
data["p1_unf_err_count2r"] = normalize_column_to_0_1(data["p1_ace_count2r"])
data["p2_unf_err_count2r"] = normalize_column_to_0_1(data["p1_ace_count2r"])

data["p1_unf_err_count2r"] = (-1) * data["p1_unf_err_count2r"]
data["p2_unf_err_count2r"] = (-1) * data["p2_unf_err_count2r"]


def normalize_to_range(column, new_min=-1, new_max=1):
    old_min = column.min()
    old_max = column.max()
    normalized_column = ((column - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return normalized_column

data["p1_Player_socre_margin"] = normalize_to_range((data["p1_Player_socre_margin"]))
data["p2_Player_socre_margin"] = normalize_to_range((data["p2_Player_socre_margin"]))

data["speed_mph"] = normalize_to_range(data["speed_mph"])
data["Players_run_distance_difference"] = normalize_to_range(data["speed_mph"])

data["p1_Players_run_distance_difference"] = normalize_to_range(data["p1_Players_run_distance_difference"])
data["p2_Players_run_distance_difference"] = normalize_to_range(data["p2_Players_run_distance_difference"])

features_weights_updated = {key.replace(' ', '_'): value for key, value in features_weights.items()}
# 假设 features_weights 已经定义，我们根据特征名前缀将其分割为两个字典
# 假设 features_weights 包含了所有特征的权重
features_weights_shared = {k.replace(' ', '_'): v for k, v in features_weights.items() if not ('p1' in k or 'p2' in k)}
features_weights_p1_specific = {k.replace(' ', '_'): v for k, v in features_weights.items() if 'p1' in k}
features_weights_p2_specific = {k.replace(' ', '_'): v for k, v in features_weights.items() if 'p2' in k}

# 将共有特征的权重加入到各自特有特征权重字典中
features_weights_p1 = {**features_weights_p1_specific, **features_weights_shared}
features_weights_p2 = {**features_weights_p2_specific, **features_weights_shared}

# 计算 p1 和 p2 的 y_value
data['y_value_p1'] = data.apply(
    lambda row: sum(features_weights_p1.get(feature, 0) * row.get(feature, 0) for feature in features_weights_p1),
    axis=1)
data['y_value_p2'] = data.apply(
    lambda row: sum(features_weights_p2.get(feature, 0) * row.get(feature, 0) for feature in features_weights_p2),
    axis=1)

data['elapsed_time'] = pd.to_timedelta(data['elapsed_time'])


# 计算归一化的y_value（如果还未计算）
def normalize_column_to_0_5(column):
    return (column - column.min()) / (column.max() - column.min()) * 35


data['y_value_p1_normalized_0_5'] = normalize_column_to_0_5(data['y_value_p1'])
data['y_value_p2_normalized_0_5'] = normalize_column_to_0_5(data['y_value_p2'])

data_subset = data.iloc[:298]

plt.figure(figsize=(10, 6))
plt.plot(data_subset['point_no'], data_subset['y_value_p1_normalized_0_5'], label='Player 1', marker='o')
plt.plot(data_subset['point_no'], data_subset['y_value_p2_normalized_0_5'], label='Player 2', marker='s')
plt.xlabel('Elapsed Time')
plt.ylabel('Normalized Y Value (0-5)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # 可能需要根据实际的时间数据调整
plt.tight_layout()  # 调整布局以适应旋转testing后的标签
plt.savefig("./output_img/动量_仅2人.png", dpi=500)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data_subset['point_no'], data_subset['p1_score'], label='Player 1', marker='o')
plt.plot(data_subset['point_no'], data_subset['p2_score'], label='Player 2', marker='s')
plt.xlabel('Elapsed Time')
plt.ylabel('Normalized Y Value (0-5)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # 可能需要根据实际的时间数据调整
plt.tight_layout()  # 调整布局以适应旋转后的标签
plt.savefig("./output_img/动量_比分.png", dpi=500)
plt.show()

import matplotlib.pyplot as plt

# 假设 data_subset 是你的数据集，已经准备好

plt.figure(figsize=(10, 6))

# Player 1 Y Value，使用蓝色的虚线表示，并调整alpha值以降低饱和度
plt.plot(data_subset['point_no'], data_subset['y_value_p1_normalized_0_5'],
         label='Player 1 Momentum Value', marker='o', linestyle='--', color='b', alpha=0.5)

# Player 2 Y Value，可以选择另一个颜色，此处以红色示例，并使用默认的alpha值
plt.plot(data_subset['point_no'], data_subset['y_value_p2_normalized_0_5'],
         label='Player 2 Momentum Value', marker='s', linestyle='--', color='r', alpha=0.5)

# Player 1 Score，使用与Y Value相同的颜色，但不透明度默认（更饱和）
plt.plot(data_subset['point_no'], data_subset['p1_score'],
         label='Player 1 Score', marker='o', color='b')

# Player 2 Score，同样保持颜色一致性
plt.plot(data_subset['point_no'], data_subset['p2_score'],
         label='Player 2 Score', marker='s', color='r')

plt.xlabel('Elapsed Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./output_img/动量_比赛.png", dpi=500)
plt.show()
data.to_csv("动量计算_完成_v1.csv", encoding="utf-8")

# %%
