import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 加载数据
# 替换下面路径为你的文件路径
file_path = '../data/cleaned_v4_logistic.csv'
data = pd.read_csv(file_path)

# 数据预处理
# 移除 'elapsed_time' 列

# 将数据分为特征（X）和目标变量（y）
X = data.drop('point_victor', axis=1)
y = data['point_victor']

# 将 'server' 列转换为哑变量
X = pd.get_dummies(X, columns=['server'], drop_first=True)

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 逻辑回归模型
log_reg = LogisticRegression(max_iter=1500)
log_reg.fit(X_train_scaled, y_train)

# 模型预测
y_pred = log_reg.predict(X_test_scaled)

# 评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix of Logistic Regression Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 预测概率
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# 计算ROC曲线，显式指定正类标签为2
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=2)

# 计算AUC值
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

#%%

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),  # 使用2阶多项式特征
    ('logistic', LogisticRegression(max_iter=1500))
])

# 训练模型
pipeline.fit(X_train_scaled, y_train)

# 进行预测
y_pred = pipeline.predict(X_test_scaled)

# 评估模型
print(classification_report(y_test, y_pred))

# 可视化ROC曲线
y_pred_prob = pipeline.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=2)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
#%%

from sklearn.metrics import roc_auc_score

# 定义多项式特征转换的度数为3
poly_features_3 = PolynomialFeatures(degree=3)

# 对训练数据和测试数据应用多项式特征转换
X_train_poly_3 = poly_features_3.fit_transform(X_train_scaled)
X_test_poly_3 = poly_features_3.transform(X_test_scaled)

# 用多项式特征训练逻辑回归模型
log_reg_poly_3 = LogisticRegression(max_iter=1500)
log_reg_poly_3.fit(X_train_poly_3, y_train)

# 对测试集进行预测
y_pred_poly_3 = log_reg_poly_3.predict(X_test_poly_3)
y_pred_prob_poly_3 = log_reg_poly_3.predict_proba(X_test_poly_3)[:, 1]

# 计算ROC曲线和AUC值
fpr_poly_3, tpr_poly_3, _ = roc_curve(y_test, y_pred_prob_poly_3, pos_label=2)
roc_auc_poly_3 = auc(fpr_poly_3, tpr_poly_3)

# 基线模型的ROC曲线和AUC值
y_pred_prob_baseline = log_reg.predict_proba(X_test_scaled)[:, 1]
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_pred_prob_baseline, pos_label=2)
roc_auc_baseline = auc(fpr_baseline, tpr_baseline)

# 二项式模型的ROC曲线和AUC值
log_reg_poly_2 = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('logistic', LogisticRegression(max_iter=1500))
])
log_reg_poly_2.fit(X_train_scaled, y_train)
y_pred_prob_poly_2 = log_reg_poly_2.predict_proba(X_test_scaled)[:, 1]
fpr_poly_2, tpr_poly_2, _ = roc_curve(y_test, y_pred_prob_poly_2, pos_label=2)
roc_auc_poly_2 = auc(fpr_poly_2, tpr_poly_2)

# 绘制ROC曲线进行比较
plt.figure(figsize=(10, 8))
plt.plot(fpr_baseline, tpr_baseline, label=f'Baseline (AUC = {roc_auc_baseline:.2f})', color='blue', lw=2)
plt.plot(fpr_poly_2, tpr_poly_2, label=f'Polynomial Degree 2 (AUC = {roc_auc_poly_2:.2f})', color='green', lw=2)
plt.plot(fpr_poly_3, tpr_poly_3, label=f'Polynomial Degree 3 (AUC = {roc_auc_poly_3:.2f})', color='red', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.savefig("./Logistic.png", dpi = 500)
plt.show()

#%%
