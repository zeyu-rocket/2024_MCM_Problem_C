import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data_path = "/Users/zeyu/Documents/Learning/数学建模/美赛数学建模/git/comap_mcm_2024/Code/Q4/data/data_analysis.csv"  # 请根据实际路径修改
data = pd.read_csv(data_path)

# 定义关键特征及其权重
key_features_weights = {
    "p1_Player_socre_margin": 8.6166,
    "p2_Player_socre_margin": 8.6166,
    "p1_1st_serve_success_ratio": 5.5833,
    "p2_1st_serve_success_ratio": 6.1,
}

# 计算原始得分
original_scores = pd.DataFrame()
for feature, weight in key_features_weights.items():
    original_scores[feature] = data[feature] * weight

# 敏感性分析 - 对每个关键特征增加10%和减少10%来观察综合得分的变化
sensitivity_results = {}
for feature in key_features_weights.keys():
    # 增加10%
    data_modified_increase = data.copy()
    data_modified_increase[feature] = data_modified_increase[feature] * 1.1
    score_increase = data_modified_increase[feature] * key_features_weights[feature]

    # 减少10%
    data_modified_decrease = data.copy()
    data_modified_decrease[feature] = data_modified_decrease[feature] * 0.9
    score_decrease = data_modified_decrease[feature] * key_features_weights[feature]

    sensitivity_results[feature] = {
        "original_mean_score": original_scores[feature].mean(),
        "increased_mean_score": score_increase.mean(),
        "decreased_mean_score": score_decrease.mean(),
    }

sensitivity_results_df = pd.DataFrame(sensitivity_results).T

# 绘制敏感性分析结果的可视化图表
plt.figure(figsize=(10, 6))

# 为每个关键特征绘制条形图
features = list(sensitivity_results_df.index)
original_scores = sensitivity_results_df["original_mean_score"].values
increased_scores = sensitivity_results_df["increased_mean_score"].values
decreased_scores = sensitivity_results_df["decreased_mean_score"].values

x = range(len(features))  # 特征标签的位置

plt.bar(x, original_scores, width=0.2, label="Original", align="center")
plt.bar(x, increased_scores, width=0.2, label="Increased 10%", align="edge")
plt.bar(
    x,
    decreased_scores,
    width=0.2,
    label="Decreased 10%",
    align="edge",
    tick_label=features,
)

plt.xlabel("Features")
plt.ylabel("Mean Score")
plt.title("Sensitivity Analysis of Key Features")
plt.xticks(x, features, rotation=45, ha="right")
plt.legend()
plt.tight_layout()

# 显示图表
plt.show()
