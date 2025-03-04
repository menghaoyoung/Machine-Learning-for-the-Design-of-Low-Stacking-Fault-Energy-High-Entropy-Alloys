# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 09:11:41 2025

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Read CSV file
df = pd.read_csv('Composition SFE Properties Dataset.csv')

# The last column is the target variable, and the remaining columns are the features
X = df.iloc[:, :-1]  # Select all columns (except for the last column)
y = df.iloc[:, -1]   # Select the last column as the target variable

# Divide the training set and testing set (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regression model (with specified parameters)
rf = RandomForestRegressor(
    n_estimators=100,         # The number of trees
    criterion='squared_error', # Evaluation criteria (mean square error)
    max_depth=3,              # Maximum Tree Depth
    min_samples_split=2,      # Minimum number of split samples
    min_samples_leaf=1,       # Minimum number of samples for leaf nodes
    random_state=42           # random seed
)

# training model
rf.fit(X_train, y_train)

# Obtain feature importance
feature_importances = rf.feature_importances_

# Create a DataFrame to save feature importance
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort in descending order of importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Save to CSV file
importance_df.to_csv('feature_importance.csv', index=False)

# Draw a feature importance bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Random Forest Feature Importance')
plt.show()

print("finish")

