import pandas as pd
data_path = "./change_points_multiple_indicators.csv"
data = pd.read_csv(data_path)

# 初始化连胜计数器
data['p1_winning_streak_count'] = 0
data['p2_winning_streak_count'] = 0
data["continuous_wins_p1"] = data["continuous_wins_p1"].replace(2,1)
data["continuous_wins_p1"] = data["continuous_wins_p1"].replace(3,1)
data["continuous_wins_p2"] = data["continuous_wins_p2"].replace(4,1)

data["continuous_wins_p2"] = data["continuous_wins_p2"].replace(2,1)
data["continuous_wins_p2"] = data["continuous_wins_p2"].replace(3,1)
data["continuous_wins_p2"] = data["continuous_wins_p2"].replace(4,1)
data.to_csv("./游程_处理完成_v1.csv", encoding="utf-8")

# max_index = int(data["Index_p2"].max())
# data["Index_p2"] = data["Index_p2"].astype(int)
# sequence = [0] * (max_index + 1)
#
# # 根据index_p2的值，在相应位置设置为1
# for index in data["Index_p2"]:
#     sequence[index] = 1
#
# # 创建DataFrame
# data = pd.DataFrame(sequence, columns=['p1_Turning_point_count'])
#
# # 展示前几行以验证结果
# data.head()
# data.to_csv("./游程_处理完成_v1.csv", encoding="utf-8")
