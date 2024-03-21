import pandas as pd

data_path = "./生成突变值检测_序列.csv"
data = pd.read_csv(data_path)

# 确定Index_p1和Index_p2中的最大值
max_index = max(data["Index_p1"].max(), data["Index_p2"].max())

# 创建一个新的DataFrame，长度为最大索引值加1，包含两个全零的序列列
new_df = pd.DataFrame({
    'p1_Turning_point_count': [0] * (max_index + 1),
    'p2_Turning_point_count': [0] * (max_index + 1)
})

# 根据Index_p1和Index_p2的值，在相应位置设置为1
for index in data["Index_p1"]:
    new_df.at[index, 'p1_Turning_point_count'] = 1

for index in data["Index_p2"]:
    new_df.at[index, 'p2_Turning_point_count'] = 1


new_df.to_csv("突变值序列数据.csv",encoding="utf-8")