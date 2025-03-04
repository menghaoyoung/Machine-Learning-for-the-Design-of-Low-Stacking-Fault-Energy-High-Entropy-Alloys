# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:31:29 2024

@author: Administrator
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global font style
plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})

# Read CSV file
data = pd.read_csv('Composition SFE Dataset.csv')

# Extract input and output data
inputs = data.iloc[:, :7]
output = data.iloc[:, 7]

# Retrieve column names for input and output as horizontal labels
input_labels = list(inputs.columns)
output_label = output.name

# Set Seaborn style
sns.set(style='whitegrid')

# Draw histograms of 7 inputs and save them
for i in range(7):
    plt.figure(figsize=(10, 6))
    sns.histplot(inputs.iloc[:, i], kde=True, color='green', edgecolor='black', linewidth=2)
    plt.title('Input {}'.format(i+1))
    plt.xlabel(input_labels[i])
    plt.ylabel('Frequency')
    plt.savefig('input_{}_histogram.png'.format(i+1), dpi=600)  # 设置dpi为600
    plt.close()

# Draw the output frequency chart and save it
plt.figure(figsize=(10, 6))
sns.histplot(output, kde=True, color='purple', edgecolor='black', linewidth=2)
plt.title(output_label)
plt.xlabel(output_label)
plt.ylabel('Frequency')
plt.savefig('output_frequency_histogram.png', dpi=600)  # 设置dpi为600
plt.close()















