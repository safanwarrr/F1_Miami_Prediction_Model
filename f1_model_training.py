#!/usr/bin/env python3
"""
F1 Model Selection and Training Script
Trains Linear Regression and Weighted Average models to predict finishing positions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Load the preprocessed data
data = pd.read_csv('preprocessed_miami_gp_data.csv')

print("=== DATA PREPARATION ===")
print(f"Total records: {len(data)}")
print(f"Records per year: {data.groupby('year').size().to_dict()}")
print(f"Missing values in avg_position_2024_pre_miami: {data['avg_position_2024_pre_miami'].isna().sum()}")
print(f"Missing values in avg_miami_position: {data['avg_miami_position'].isna().sum()}")

# Create training dataset using 2022-2023 data to predict 2024
train_data = data[data['year'].isin([2022, 2023])].copy()
test_data = data[data['year'] == 2024].copy()

# Fill missing values for training
train_data['avg_position_2024_pre_miami'] = train_data['avg_position_2024_pre_miami'].fillna(train_data['position'])
train_data['avg_miami_position'] = train_data['avg_miami_position'].fillna(train_data['position'])

# Fill missing values for test data
test_data['avg_position_2024_pre_miami'] = test_data['avg_position_2024_pre_miami'].fillna(test_data['position'])
test_data['avg_miami_position'] = test_data['avg_miami_position'].fillna(test_data['position'])

# Prepare features and target for training
features = ['avg_position_2024_pre_miami', 'avg_miami_position', 'grid_position', 'avg_points_per_race']
X_train = train_data[features]
y_train = train_data['position']

# Prepare test data
X_test = test_data[features]
y_test = test_data['position']

print(f"\nTraining data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# ====================================
# Simple Linear Regression Model
# ====================================
print("\n=== LINEAR REGRESSION MODEL ===")

# Train linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions on test set
y_pred_linear = linear_model.predict(X_test)

# Calculate metrics
linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))
linear_mae = mean_absolute_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

print(f"Linear Regression Results:")
print(f"  RMSE: {linear_rmse:.3f}")
print(f"  MAE: {linear_mae:.3f}")
print(f"  R²: {linear_r2:.3f}")

# Feature importance
print(f"\nFeature coefficients:")
for feature, coef in zip(features, linear_model.coef_):
    print(f"  {feature}: {coef:.3f}")
print(f"  Intercept: {linear_model.intercept_:.3f}")

# ====================================
# Weighted Average Model
# ====================================
print("\n=== WEIGHTED AVERAGE MODEL ===")

def weighted_average(row):
    """Calculate weighted average of previous race positions and historical Miami performance"""
    # Weight: 60% historical Miami performance, 30% pre-Miami 2024, 10% grid position
    miami_weight = 0.6 if not pd.isna(row['avg_miami_position']) else 0.0
    pre_miami_weight = 0.3 if not pd.isna(row['avg_position_2024_pre_miami']) else 0.0
    grid_weight = 0.1
    
    # Normalize weights if some data is missing
    total_weight = miami_weight + pre_miami_weight + grid_weight
    
    if total_weight == 0:
        return row['grid_position']  # Fallback to grid position
    
    prediction = 0
    if miami_weight > 0:
        prediction += (miami_weight / total_weight) * row['avg_miami_position']
    if pre_miami_weight > 0:
        prediction += (pre_miami_weight / total_weight) * row['avg_position_2024_pre_miami']
    prediction += (grid_weight / total_weight) * row['grid_position']
    
    return prediction

# Predict using weighted average method
y_pred_weighted = X_test.apply(weighted_average, axis=1)

# Calculate metrics
weighted_rmse = np.sqrt(mean_squared_error(y_test, y_pred_weighted))
weighted_mae = mean_absolute_error(y_test, y_pred_weighted)
weighted_r2 = r2_score(y_test, y_pred_weighted)

print(f"Weighted Average Results:")
print(f"  RMSE: {weighted_rmse:.3f}")
print(f"  MAE: {weighted_mae:.3f}")
print(f"  R²: {weighted_r2:.3f}")

# ====================================
# Cross-Validation on Training Data
# ====================================
print("\n=== CROSS-VALIDATION ON TRAINING DATA ===")

# Use cross-validation on training data only
linear_cv_scores = cross_val_score(linear_model, X_train, y_train, 
                                  cv=5, scoring='neg_mean_squared_error')
linear_cv_rmse = np.sqrt(-linear_cv_scores)

print(f"Linear Regression CV Results:")
print(f"  Average RMSE: {np.mean(linear_cv_rmse):.3f} (+/- {np.std(linear_cv_rmse)*2:.3f})")

# Manual cross-validation for weighted average
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
weighted_cv_rmse = []

for train_idx, val_idx in kf.split(X_train):
    X_val_cv = X_train.iloc[val_idx]
    y_val_cv = y_train.iloc[val_idx]
    
    y_pred_weighted_cv = X_val_cv.apply(weighted_average, axis=1)
    cv_rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_weighted_cv))
    weighted_cv_rmse.append(cv_rmse)

print(f"Weighted Average CV Results:")
print(f"  Average RMSE: {np.mean(weighted_cv_rmse):.3f} (+/- {np.std(weighted_cv_rmse)*2:.3f})")

# ====================================
# Results Comparison and Predictions
# ====================================
print("\n=== MODEL COMPARISON ===")
print(f"Linear Regression Test RMSE: {linear_rmse:.3f}")
print(f"Weighted Average Test RMSE: {weighted_rmse:.3f}")

if linear_rmse < weighted_rmse:
    print("\n✓ Linear Regression performs better")
    best_model = "Linear Regression"
    best_predictions = y_pred_linear
else:
    print("\n✓ Weighted Average performs better")
    best_model = "Weighted Average"
    best_predictions = y_pred_weighted

# ====================================
# Detailed Predictions for 2024 Miami GP
# ====================================
print("\n=== 2024 MIAMI GP PREDICTIONS ===")
print(f"Using {best_model} model:\n")

# Create predictions dataframe
predictions_df = test_data[['driver_name', 'team', 'position']].copy()
predictions_df['predicted_position_linear'] = y_pred_linear
predictions_df['predicted_position_weighted'] = y_pred_weighted
predictions_df['actual_position'] = y_test

# Sort by predicted position (using best model)
if best_model == "Linear Regression":
    predictions_df = predictions_df.sort_values('predicted_position_linear')
else:
    predictions_df = predictions_df.sort_values('predicted_position_weighted')

print("Driver                 | Team           | Predicted | Actual | Error")
print("-" * 70)
for _, row in predictions_df.iterrows():
    pred_pos = row['predicted_position_linear'] if best_model == "Linear Regression" else row['predicted_position_weighted']
    error = abs(pred_pos - row['actual_position'])
    print(f"{row['driver_name']:<20} | {row['team']:<13} | {pred_pos:8.1f} | {row['actual_position']:6.0f} | {error:5.1f}")

# Save predictions
predictions_df.to_csv('miami_gp_2024_predictions.csv', index=False)
print(f"\nPredictions saved to 'miami_gp_2024_predictions.csv'")

print(f"\n=== TRAINING COMPLETE ===")
print(f"Best performing model: {best_model}")
print(f"Best model RMSE: {min(linear_rmse, weighted_rmse):.3f}")
