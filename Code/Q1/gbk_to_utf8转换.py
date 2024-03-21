import pandas as pd

gpt_input_cleaning_rule = pd.read_csv("../../Gpt_Input_Materials/Data/Data_cleaning_rule.csv", encoding="gbk")
gpt_input_cleaning_rule.to_csv("../../Gpt_Input_Materials/Data/Data_cleaning_rule_utf8.csv", encoding="utf-8")
#%%
