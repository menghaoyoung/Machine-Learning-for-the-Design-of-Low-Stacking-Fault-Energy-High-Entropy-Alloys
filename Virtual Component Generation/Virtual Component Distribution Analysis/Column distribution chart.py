# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:30:44 2024

@author: Administrator
"""


import pandas as pd
import matplotlib.pyplot as plt

# Read data from the CSV file provided by the user
def read_csv(file_path):
    return pd.read_csv(file_path)

# Draw a bar chart of content distribution
def plot_distribution(data):
    for col in data.columns:
        plt.figure()
        plt.hist(data[col], bins=10, color='green', edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel('Content Range')
        plt.ylabel('Frequency')
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)  # Adjust the style, color, and thickness of the background guide lines
        plt.savefig(f'{col}_distribution.png', dpi=900)  # Save image and set dpi
        plt.show()

# CSV file path provided by the user
csv_file_path = "alloy_compositions.csv"

# read data
data = read_csv(csv_file_path)

# Draw a bar chart of the element content distribution and save the image
plot_distribution(data)


