import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

feature_output_path = "./feature_rank"

# 生成模拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化模型列表
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Random Forest', RandomForestClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('GBM', GradientBoostingClassifier()),
    ('LightGBM', LGBMClassifier()),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('SVM', SVC(probability=True))
]

# 对数据进行标准化处理，SVM在标准化数据上表现更好
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))

for name, model in models:
    if name == 'SVM':  # 对于SVM使用标准化数据
        model.fit(X_train_scaled, y_train)
        y_scores = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_scores = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()

# 特征名称
feature_names = [f'Feature {i}' for i in range(X.shape[1])]

for name, model in models:
    # 训练模型
    if name == 'SVM':  # 对于SVM使用标准化数据
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)

    # 如果模型有feature_importances_属性，则保存特征重要度
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df = df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        df.to_csv(feature_output_path + "/" + f'{name}_feature_importances.csv', index=False)
        print(f'{name} feature importances saved to {name}_feature_importances.csv')
    # 对于逻辑回归，使用系数作为特征重要度
    elif name == 'Logistic Regression':
        coefficients = model.coef_.flatten()
        df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        df = df.sort_values(by='Coefficient', ascending=False).reset_index(drop=True)
        df.to_csv(feature_output_path + "/" + f'{name}_coefficients.csv', index=False)
        print(f'{name} coefficients saved to {name}_coefficients.csv')
# %%
