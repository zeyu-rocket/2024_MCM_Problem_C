from datetime import datetime
from datetime import time
import pandas as pd
from sklearn.preprocessing import StandardScaler

data_path = "lstm_gpt_input.xlsx"
data = pd.read_excel(data_path)

# 数据标准化
columns_to_scale = ["y_value_p1", "y_value_p2"]
scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
data_scaled[columns_to_scale].head()

