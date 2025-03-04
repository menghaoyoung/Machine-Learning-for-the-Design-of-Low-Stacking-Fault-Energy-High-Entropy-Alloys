# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 17:24:42 2025

@author: Administrator
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv('Composition SFE Properties Dataset.csv')

# Calculate Pearson correlation coefficient
correlation_matrix = df.corr(method='pearson')

# Save the correlation coefficient matrix as a CSV file
correlation_matrix.to_csv('correlation_matrix.csv', index=True)

# Draw a heatmap (without displaying numerical labels)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5, annot=False)  # annot=False 隐藏数值
plt.title('Pearson Correlation Heatmap')
plt.show()

print("finish")
