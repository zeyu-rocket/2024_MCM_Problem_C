import pandas as pd
from sklearn.preprocessing import StandardScaler

sample_data = pd.read_csv('../data/Wimbledon_featured_matches.csv')

# 特殊值替换和范围检查
sample_data['p1_score'].replace({'AD': 60}, inplace=True)
sample_data['p2_score'].replace({'AD': 60}, inplace=True)
sample_data['p1_score'] = pd.to_numeric(sample_data['p1_score'], errors='coerce')
sample_data['p2_score'] = pd.to_numeric(sample_data['p2_score'], errors='coerce')

sample_data['winner_shot_type'].replace({'F': 1}, inplace=True)
sample_data['winner_shot_type'].replace({'B': 2}, inplace=True)
sample_data['winner_shot_type'] = pd.to_numeric(sample_data['winner_shot_type'], errors='coerce')
sample_data['winner_shot_type'] = pd.to_numeric(sample_data['winner_shot_type'], errors='coerce')

valid_scores = [0, 15, 30, 40, 60]
sample_data = sample_data[sample_data['p1_score'].isin(valid_scores) & sample_data['p2_score'].isin(valid_scores)]

sample_data = sample_data[(sample_data['p1_distance_run'] > 0) & (sample_data['p2_distance_run'] > 0)]

# 变量变更
serve_width_mapping = {'B': 1, 'BC': 2, 'BW': 3, 'C': 4, 'W': 5}
serve_depth_mapping = {'CTL': 1, 'NCTL': 2}
return_depth_mapping = {'D': 1, 'ND': 2}

sample_data['serve_width'] = sample_data['serve_width'].map(serve_width_mapping)
sample_data['serve_depth'] = sample_data['serve_depth'].map(serve_depth_mapping)
sample_data['return_depth'] = sample_data['return_depth'].map(return_depth_mapping)
columns_to_check = ['speed_mph']
# 循环检查每列，如果存在NA则删除该列
for column in columns_to_check:
    if sample_data[column].isna().any():  # 如果列中存在任何NA值
        sample_data = sample_data.dropna(subset=[column])  # 删除包含NA的行

# 输出清洗后的数据
print(sample_data.head())

sample_data.reset_index(drop=True, inplace=True)
# 初始化连续胜利计数器
sample_data['continuous_wins_p1'] = 0
sample_data['continuous_wins_p2'] = 0

# 遍历数据，从第二行开始
for i in range(1, len(sample_data)):
    # 获取当前行和前一行的分数
    p1_prev = sample_data.iloc[i - 1]['p1_score']
    p2_prev = sample_data.iloc[i - 1]['p2_score']
    p1_curr = sample_data.iloc[i]['p1_score']
    p2_curr = sample_data.iloc[i]['p2_score']

    # 检查是否连续赢球的情况
    if p1_curr > p1_prev and p2_curr == p2_prev:
        # 球员1连续赢球
        sample_data.loc[i, 'continuous_wins_p1'] = sample_data.loc[i - 1, 'continuous_wins_p1'] + 1
        sample_data.loc[i, 'continuous_wins_p2'] = 0
    elif p2_curr > p2_prev and p1_curr == p1_prev:
        # 球员2连续赢球
        sample_data.loc[i, 'continuous_wins_p2'] = sample_data.loc[i - 1, 'continuous_wins_p2'] + 1
        sample_data.loc[i, 'continuous_wins_p1'] = 0
    else:
        # 如果没有连续赢球，则重置计数器
        sample_data.loc[i, 'continuous_wins_p1'] = 0
        sample_data.loc[i, 'continuous_wins_p2'] = 0

# 确保第一行的计数器也正确初始化
sample_data.loc[0, 'continuous_wins_p1'] = 0
sample_data.loc[0, 'continuous_wins_p2'] = 0

sample_data["Player_socre_margin"] = sample_data["p1_points_won"] - sample_data["p2_points_won"]
sample_data["Players_run_distance_difference"] = sample_data["p1_distance_run"] - sample_data["p2_distance_run"]

# 构建一发成功率 一发得分率 二发得分率
# 初始化列
sample_data['p1_first_serve_success_count'] = 0
sample_data['p1_first_serve_total_count'] = 0
sample_data['p1_first_serve_win_count'] = 0
sample_data['p1_second_serve_win_count'] = 0
sample_data['p1_second_serve_total_count'] = 0

sample_data['p2_first_serve_success_count'] = 0
sample_data['p2_first_serve_total_count'] = 0
sample_data['p2_first_serve_win_count'] = 0
sample_data['p2_second_serve_win_count'] = 0
sample_data['p2_second_serve_total_count'] = 0

p1_first_successful_count = 0
p1_first_serve_total_count = 0
p1_first_serve_win_count = 0
p1_second_serve_total_count = 0
p1_second_serve_win_count = 0

p2_first_successful_count = 0
p2_first_serve_total_count = 0
p2_first_serve_win_count = 0
p2_second_serve_total_count = 0
p2_second_serve_win_count = 0
current_match_id = sample_data['match_id'][0]

for i in range(len(sample_data)):
    if sample_data.loc[i, 'match_id'] != current_match_id:
        # 重置计数器
        p1_first_successful_count = 0
        p1_first_serve_total_count = 0
        p1_first_serve_win_count = 0
        p1_second_serve_total_count = 0
        p1_second_serve_win_count = 0

        p2_first_successful_count = 0
        p2_first_serve_total_count = 0
        p2_first_serve_win_count = 0
        p2_second_serve_total_count = 0
        p2_second_serve_win_count = 0
        current_match_id = sample_data.loc[i, 'match_id']

    if sample_data.loc[i, 'serve_no'] == 1 and sample_data.loc[i, 'server'] == 1:
        p1_first_serve_total_count += 1
        if pd.notnull(sample_data.loc[i, 'return_depth']):
            p1_first_successful_count += 1
            if sample_data.loc[i, 'server'] == sample_data.loc[i, 'point_victor']:
                p1_first_serve_win_count += 1

    elif sample_data.loc[i, 'serve_no'] == 2 and sample_data.loc[i, 'server'] == 1:
        p1_second_serve_total_count += 1
        if pd.notnull(sample_data.loc[i, 'return_depth']) and sample_data.loc[i, 'server'] == sample_data.loc[
            i, 'point_victor']:
            p1_second_serve_win_count += 1

    if sample_data.loc[i, 'serve_no'] == 1 and sample_data.loc[i, 'server'] == 2:
        p2_first_serve_total_count += 1
        if pd.notnull(sample_data.loc[i, 'return_depth']):
            p2_first_successful_count += 1
            if sample_data.loc[i, 'server'] == sample_data.loc[i, 'point_victor']:
                p2_first_serve_win_count += 1

    elif sample_data.loc[i, 'serve_no'] == 2 and sample_data.loc[i, 'server'] == 2:
        p2_second_serve_total_count += 1
        if pd.notnull(sample_data.loc[i, 'return_depth']) and sample_data.loc[i, 'server'] == sample_data.loc[
            i, 'point_victor']:
            p2_second_serve_win_count += 1

    # 更新DataFrame
    sample_data.at[i, 'p1_first_serve_success_count'] = p1_first_successful_count
    sample_data.at[i, 'p1_first_serve_total_count'] = p1_first_serve_total_count
    sample_data.at[i, 'p1_first_serve_win_count'] = p1_first_serve_win_count
    sample_data.at[i, 'p1_second_serve_win_count'] = p1_second_serve_win_count
    sample_data.at[i, 'p1_second_serve_total_count'] = p1_second_serve_total_count

    sample_data.at[i, 'p2_first_serve_success_count'] = p2_first_successful_count
    sample_data.at[i, 'p2_first_serve_total_count'] = p2_first_serve_total_count
    sample_data.at[i, 'p2_first_serve_win_count'] = p2_first_serve_win_count
    sample_data.at[i, 'p2_second_serve_win_count'] = p2_second_serve_win_count
    sample_data.at[i, 'p2_second_serve_total_count'] = p2_second_serve_total_count

# 计算比率
sample_data["p1_1st_serve_success_ratio"] = sample_data['p1_first_serve_success_count'] / sample_data[
    'p1_first_serve_total_count']
sample_data["p1_1st_serve_win_ratio"] = sample_data['p1_first_serve_win_count'] / sample_data[
    'p1_first_serve_total_count']
sample_data["p1_2nd_serve_win_ratio"] = sample_data['p1_second_serve_win_count'] / sample_data[
    'p1_second_serve_total_count']  # 注意这里可能需要调整，根据实际逻辑

sample_data["p2_1st_serve_success_ratio"] = sample_data['p2_first_serve_success_count'] / sample_data[
    'p2_first_serve_total_count']
sample_data["p2_1st_serve_win_ratio"] = sample_data['p2_first_serve_win_count'] / sample_data[
    'p2_first_serve_total_count']
sample_data["p2_2nd_serve_win_ratio"] = sample_data['p2_second_serve_win_count'] / sample_data[
    'p2_second_serve_total_count']  # 注意这里可能需要调整，根据实际逻辑

columns_to_initialize = [
    'p1_first_serve_success_count',
    'p1_first_serve_total_count',
    'p1_first_serve_win_count',
    'p1_second_serve_win_count',
    'p1_second_serve_total_count',
    'p2_first_serve_success_count',
    'p2_first_serve_total_count',
    'p2_first_serve_win_count',
    'p2_second_serve_win_count',
    'p2_second_serve_total_count',
    'p1_1st_serve_success_ratio',
    'p1_1st_serve_win_ratio',
    'p1_2nd_serve_win_ratio',
    'p2_1st_serve_success_ratio',
    'p2_1st_serve_win_ratio',
    'p2_2nd_serve_win_ratio'
]
sample_data[columns_to_initialize] = sample_data[columns_to_initialize].fillna(0)

ace_counts = [sample_data['p1_ace'].iloc[max(i - 2, 0):i + 1].sum() for i in range(len(sample_data))]
sample_data['p1_ace_count2r'] = ace_counts

p2_ace_count2 = [sample_data['p2_ace'].iloc[max(i - 2, 0):i + 1].sum() for i in range(len(sample_data))]
sample_data['p2_ace_count2r'] = p2_ace_count2

p1_double_fault_count2r = [sample_data['p1_double_fault'].iloc[max(i - 2, 0):i + 1].sum() for i in
                           range(len(sample_data))]
sample_data['p1_double_fault_count2r'] = p1_double_fault_count2r

p2_double_fault_count2r = [sample_data['p2_double_fault'].iloc[max(i - 2, 0):i + 1].sum() for i in
                           range(len(sample_data))]
sample_data['p2_double_fault_count2r'] = p2_double_fault_count2r

p1_unf_err_count2r = [sample_data['p1_unf_err'].iloc[max(i - 2, 0):i + 1].sum() for i in range(len(sample_data))]
sample_data['p1_unf_err_count2r'] = p1_unf_err_count2r

p2_unf_err_count2r = [sample_data['p2_unf_err'].iloc[max(i - 2, 0):i + 1].sum() for i in range(len(sample_data))]
sample_data['p2_unf_err_count2r'] = p2_unf_err_count2r

sample_data['s_change'] = 0  # 先将所有变化标记为0

# 检测变化
for i in range(1, len(sample_data)):
    # 检测基本变化
    if sample_data.loc[i, "return_depth"] != sample_data.loc[i - 1, "return_depth"]:
        sample_data.loc[i, 's_change'] = 1

    # 检测连续变化
    if i >= 2 and sample_data.loc[i, "return_depth"] != sample_data.loc[i - 2, "return_depth"] and sample_data.loc[i - 1, "return_depth"] != sample_data.loc[i - 2, "return_depth"]:
        sample_data.loc[i, 's_change'] = 2

sample_data.to_csv("../data/cleaned_v1_没有标准化.csv", encoding="utf-8")
sample_data["Player_socre_margin"] = sample_data["p1_points_won"] - sample_data["p2_points_won"]
sample_data["Players_run_distance_difference"] = sample_data["p1_distance_run"] - sample_data["p2_distance_run"]

# 加载数据
# 选择要标准化的列
columns_to_scale = [
    'p1_score', 'p2_score', 'p1_points_won', 'p2_points_won',
    'p1_distance_run', 'p2_distance_run', 'rally_count', 'speed_mph',
    'serve_width', 'serve_depth', 'return_depth', 'Player_socre_margin', 'Players_run_distance_difference',
    'p1_1st_serve_success_ratio', 'p1_1st_serve_win_ratio', 'p1_2nd_serve_win_ratio', 'p2_1st_serve_success_ratio',
    'p2_1st_serve_win_ratio', 'p2_2nd_serve_win_ratio',
    'p1_ace_count2r', 'p2_ace_count2r', 'p1_double_fault_count2r', 'p2_double_fault_count2r', 'p1_unf_err_count2r',
    'p2_unf_err_count2r'
]

# 实例化标准化器
scaler = StandardScaler()

# 对选定的列进行标准化
data_scaled = sample_data.copy()
data_scaled[columns_to_scale] = scaler.fit_transform(sample_data[columns_to_scale])

# 检查前几行以确认标准化效果
data_scaled[columns_to_scale].head()
data_scaled.to_csv('../data/cleaned_v1.csv', index=False, encoding="utf-8")

# %%
