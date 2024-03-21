import pandas as pd
import numpy as np

# 读取比赛数据
match_data_path = "/Users/zeyu/Documents/Learning/数学建模/美赛数学建模/git/comap_mcm_2024/Code/Q3/data/2023-wimbledon-1301_match_data.csv"
match_data = pd.read_csv(match_data_path)

# 将p1_score和p2_score转换为数值类型，非数值转换为NaN后填充为0
match_data["p1_score"] = pd.to_numeric(match_data["p1_score"], errors="coerce")
match_data["p2_score"] = pd.to_numeric(match_data["p2_score"], errors="coerce")
match_data.fillna(0, inplace=True)

# 计算p1_score和p2_score的平均值
match_data["score_avg"] = (match_data["p1_score"] + match_data["p2_score"]) / 2

# 准备数据以适用于LSTM模型，格式为[samples, time steps, features]
# 这里每行视作一个时间步，'score_avg'作为特征
lstm_data = match_data["score_avg"].values.reshape((match_data.shape[0], 1, 1))

# 将LSTM数据转换为2D格式以保存到CSV文件
lstm_data_2d = lstm_data.reshape(-1, 1)  # 假设每个样本只有一个特征
lstm_df = pd.DataFrame(lstm_data_2d, columns=["score_avg"])

# 指定输出文件路径
output_path = "/Users/zeyu/Documents/Learning/数学建模/美赛数学建模/git/comap_mcm_2024/Code/Q3/data/2023-wimbledon-1301_lstm_data.csv"

# 保存DataFrame到CSV文件
lstm_df.to_csv(output_path, index=False)

print(f"Processed LSTM data saved to {output_path}")
