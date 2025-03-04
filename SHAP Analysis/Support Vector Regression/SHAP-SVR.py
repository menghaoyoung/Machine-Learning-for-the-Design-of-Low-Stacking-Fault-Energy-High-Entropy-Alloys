# -*- coding: utf-8 -*-
"""

@author: Administrator
"""

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # Import SVR

# Read CSV file
file_path = 'filtered_data.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Separate features (alloy elements) and target performance
X = df[['Fe', 'Mn', 'Ni', 'Co', 'Cr','V','Al']]  # Replace with your alloy element column name
y = df['SVR-prediction']  # Replace with your target performance column name

# Divide the training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using SVR model and setting hyperparameters
model = SVR(
    C=866.1762795987894,  # Set C parameter
    kernel='rbf'  # Using radial basis kernel function
)
model.fit(X_train, y_train)

# Use shap.sample to reduce the sample size of background data
background_data = shap.sample(X_train, 100)  # Only use 100 background samples

# Calculate SHAP value using KernelExplainer
explainer = shap.KernelExplainer(model.predict, background_data, link="identity")
shap_values = explainer.shap_values(X_test, nsamples=100)  # Control nsamples to reduce computation time

# Save SHAP values to CSV file
shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
shap_values_df.to_csv('shap_values.csv', index=False)

# 1. SHAP Summary Chart - Bar Chart
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X.columns, sort=False)
plt.title("SHAP Summary Bar Plot")
plt.savefig('SHAP/shap_summary_bar_plot.png')
plt.show()

# 2. SHAP Summary Chart - Scatter Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, sort=False)
plt.title("SHAP Summary Dot Plot")
plt.savefig('SHAP/shap_summary_dot_plot.png')
plt.show()

# 4. SHAP Force Plot - Plot (showing the SHAP values of a single sample)
#Only display the first sample's plot as an example
plt.figure(figsize=(10, 6))
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True)
plt.title("SHAP Force Plot for a Single Sample")
plt.savefig('SHAP/shap_force_plot_sample.png')
plt.show()
