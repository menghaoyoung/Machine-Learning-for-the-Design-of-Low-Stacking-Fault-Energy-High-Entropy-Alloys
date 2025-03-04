# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:30:44 2024

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
data = pd.read_csv('Composition SFE Dataset.csv')

# Get input and output column names
input_columns = data.columns[:-1]  # The first 7 columns are inputs
output_column = data.columns[-1]    # The last column is the output

# Draw scatter plots for each input and output, and output them to a file
for input_column in input_columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(data[input_column], data[output_column], color='green', alpha=0.5)
    plt.title(f'{output_column} vs {input_column}')
    plt.xlabel(input_column)
    plt.ylabel(output_column)
    plt.grid(True)
    plt.savefig(f'{output_column}_vs_{input_column}.png', dpi=900)  # Output graphics to a file and set DPI to 900
    plt.close()  # Close the current graphic to draw the next one

print("The image has been saved.")

