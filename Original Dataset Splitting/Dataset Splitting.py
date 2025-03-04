# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:50:36 2024

@author: Administrator
"""

import pandas as pd
import random

def select_random_data(df, num_to_select):
    if num_to_select > len(df):
        raise ValueError("要选择的数据量超过了数据集的大小！")
    selected_indices = random.sample(range(len(df)), num_to_select)
    selected_data = df.iloc[selected_indices]
    remaining_data = df.drop(index=selected_indices)
    return selected_data, remaining_data

# 从csv文件中读取数据集
file_path = "Composition SFE Properties Dataset.csv"  # 请将路径替换为你的数据集路径
user_dataset = pd.read_csv(file_path)

# 随机选择100组数据
num_to_select = 100
selected_data, remaining_data = select_random_data(user_dataset, num_to_select)

# 保存选取的数据集和剩余的数据集为CSV文件
selected_data.to_csv("test_data.csv", index=False)
remaining_data.to_csv("train_data.csv", index=False)

print("选取的数据集已保存为 test_data.csv")
print("剩余的数据集已保存为 train_data.csv")

