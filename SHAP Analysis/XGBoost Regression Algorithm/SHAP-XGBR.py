# -*- coding: utf-8 -*-
"""

@author: Administrator
"""

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # Import XGBRegressor
import os

# Read CSV file
file_path = 'filtered_data.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Separate features (alloy elements) and target performance
X = df[['Fe', 'Mn', 'Ni', 'Co', 'Cr','V','Al']]  # Replace with your alloy element column name
y = df['XGB-prediction']  # Replace with your target performance column name

# Divide the training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using XGBRegressor model and setting hyperparameters
model = XGBRegressor(
    colsample_bytree=0.7565404620534099,
    learning_rate=0.182237522488218,
    max_depth=3,
    n_estimators=55,
    subsample=0.670800178345588,
    booster='gbtree',  # Use the specified base estimator
    random_state=42
)
model.fit(X_train, y_train)

#Calculate SHAP value
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Save SHAP values to CSV file
shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
shap_values_df.to_csv('shap_values.csv', index=False)

# Create SHAP directory to save images
os.makedirs('SHAP', exist_ok=True)

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
shap.force_plot(explainer.expected_value, shap_values.values[0, :], X_test.iloc[0, :], matplotlib=True)
plt.title("SHAP Force Plot for a Single Sample")
plt.savefig('SHAP/shap_force_plot_sample.png')
plt.show()
