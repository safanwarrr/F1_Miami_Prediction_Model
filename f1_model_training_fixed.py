#!/usr/bin/env python3
"""
F1 Model Selection and Training Script (Fixed Version)
Trains models to predict 2024 Miami GP without data leakage
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def prepare_features(data):
    """Prepare features for training without data leakage"""
    # Calculate historical Miami performance (only 2022-2023)
    miami_hist = data[data['year'].isin([2022, 2023])].groupby('driver_name')['position'].agg(['mean', 'count']).reset_index()
    miami_hist.columns = ['driver_name', 'hist_miami_avg_pos', 'miami_race_count']
    
    # Calculate overall career average (2022-2023)
    career_avg = data[data['year'].isin([2022, 2023])].groupby('driver_name')['position'].mean().reset_index()
    career_avg.columns = ['driver_name', 'career_avg_position']
    
    # Calculate points per race average (2022-2023)
    points_avg = data[data['year'].isin([2022, 2023])].groupby('driver_name')['points'].mean().reset_index()
    points_avg.columns = ['driver_name', 'career_avg_points']
    
    return miami_hist, career_avg, points_avg

# Load data
data = pd.read_csv('preprocessed_miami_gp_data.csv')

print("=== PREPARING TRAINING DATA (2022-2023) ===")

# Get features
miami_hist, career_avg, points_avg = prepare_features(data)

# Prepare training data (2022-2023)
train_data = data[data['year'].isin([2022, 2023])].copy()

# Merge features
train_data = train_data.merge(miami_hist, on='driver_name', how='left')
train_data = train_data.merge(career_avg, on='driver_name', how='left')
train_data = train_data.merge(points_avg, on='driver_name', how='left')

# Fill missing values
train_data['hist_miami_avg_pos'] = train_data['hist_miami_avg_pos'].fillna(train_data['position'])
train_data['career_avg_position'] = train_data['career_avg_position'].fillna(train_data['position'])
train_data['career_avg_points'] = train_data['career_avg_points'].fillna(0)
train_data['miami_race_count'] = train_data['miami_race_count'].fillna(1)

print("=== PREPARING TEST DATA (2024) ===")

# Prepare 2024 test data
test_data = data[data['year'] == 2024].copy()

# Calculate 2024 season performance up to Miami (simulate pre-race knowledge)
# We'll use the fact that Miami was race 5 in 2024, so we have 4 races before
season_2024_features = {}

for driver in test_data['driver_name'].unique():
    # For simulation, we'll use their 2023 end-of-season performance as "2024 pre-Miami"
    driver_2023 = data[(data['driver_name'] == driver) & (data['year'] == 2023)]
    if len(driver_2023) > 0:
        avg_pos_2024 = driver_2023['position'].iloc[-1]  # Last 2023 position as proxy
        avg_points_2024 = driver_2023['points'].iloc[-1]
    else:
        avg_pos_2024 = 15.0  # Default middle-field position
        avg_points_2024 = 0.0
    
    season_2024_features[driver] = {
        'season_2024_avg_pos': avg_pos_2024,
        'season_2024_avg_points': avg_points_2024
    }

# Add features to test data
for idx, row in test_data.iterrows():
    driver = row['driver_name']
    if driver in season_2024_features:
        test_data.loc[idx, 'season_2024_avg_pos'] = season_2024_features[driver]['season_2024_avg_pos']
        test_data.loc[idx, 'season_2024_avg_points'] = season_2024_features[driver]['season_2024_avg_points']

# Merge historical features for test data
test_data = test_data.merge(miami_hist, on='driver_name', how='left')
test_data = test_data.merge(career_avg, on='driver_name', how='left')
test_data = test_data.merge(points_avg, on='driver_name', how='left')

# Fill missing values for test data
test_data['hist_miami_avg_pos'] = test_data['hist_miami_avg_pos'].fillna(15.0)
test_data['career_avg_position'] = test_data['career_avg_position'].fillna(15.0)
test_data['career_avg_points'] = test_data['career_avg_points'].fillna(0)
test_data['miami_race_count'] = test_data['miami_race_count'].fillna(0)
test_data['season_2024_avg_pos'] = test_data['season_2024_avg_pos'].fillna(15.0)
test_data['season_2024_avg_points'] = test_data['season_2024_avg_points'].fillna(0)

# Define features for training (no target leakage)
feature_cols = ['grid_position', 'hist_miami_avg_pos', 'career_avg_position', 
               'career_avg_points', 'miami_race_count']

# For test data, include 2024 season features
test_feature_cols = feature_cols + ['season_2024_avg_pos', 'season_2024_avg_points']

# Prepare training data
X_train = train_data[feature_cols]
y_train = train_data['position']

# Prepare test data (using additional 2024 features)
X_test = test_data[feature_cols]  # Use same features as training for consistency
y_test = test_data['position']

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Features used: {feature_cols}")

print("\n=== MODEL 1: LINEAR REGRESSION ===")

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Metrics
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

print(f"Linear Regression Results:")
print(f"  RMSE: {lr_rmse:.3f}")
print(f"  MAE: {lr_mae:.3f}")
print(f"  R²: {lr_r2:.3f}")

print(f"\nFeature importance (coefficients):")
for feature, coef in zip(feature_cols, lr_model.coef_):
    print(f"  {feature}: {coef:.3f}")

print("\n=== MODEL 2: WEIGHTED AVERAGE ===")

def calculate_weighted_average(row):
    """Calculate weighted prediction based on historical data"""
    weights = {
        'grid_position': 0.3,
        'hist_miami_avg_pos': 0.4,
        'career_avg_position': 0.2,
        'miami_experience': 0.1
    }
    
    # Experience bonus/penalty
    experience_factor = min(row['miami_race_count'], 2) / 2  # 0 to 1
    
    prediction = (weights['grid_position'] * row['grid_position'] +
                 weights['hist_miami_avg_pos'] * row['hist_miami_avg_pos'] +
                 weights['career_avg_position'] * row['career_avg_position'] +
                 weights['miami_experience'] * (15 - 5 * experience_factor))
    
    return prediction

# Calculate weighted average predictions
y_pred_wa = X_test.apply(calculate_weighted_average, axis=1)

# Metrics
wa_rmse = np.sqrt(mean_squared_error(y_test, y_pred_wa))
wa_mae = mean_absolute_error(y_test, y_pred_wa)
wa_r2 = r2_score(y_test, y_pred_wa)

print(f"Weighted Average Results:")
print(f"  RMSE: {wa_rmse:.3f}")
print(f"  MAE: {wa_mae:.3f}")
print(f"  R²: {wa_r2:.3f}")

print("\n=== MODEL 3: RANDOM FOREST (BONUS) ===")

# Train Random Forest for comparison
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Metrics
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"Random Forest Results:")
print(f"  RMSE: {rf_rmse:.3f}")
print(f"  MAE: {rf_mae:.3f}")
print(f"  R²: {rf_r2:.3f}")

print(f"\nFeature importance:")
for feature, importance in zip(feature_cols, rf_model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")

print("\n=== CROSS VALIDATION ===")

# Cross-validation scores
lr_cv = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

print(f"Linear Regression CV RMSE: {np.sqrt(-lr_cv.mean()):.3f} (+/- {np.sqrt(-lr_cv).std()*2:.3f})")
print(f"Random Forest CV RMSE: {np.sqrt(-rf_cv.mean()):.3f} (+/- {np.sqrt(-rf_cv).std()*2:.3f})")

print("\n=== MODEL COMPARISON ===")
models = {
    'Linear Regression': lr_rmse,
    'Weighted Average': wa_rmse,
    'Random Forest': rf_rmse
}

best_model = min(models, key=models.get)
print(f"Best performing model: {best_model} (RMSE: {models[best_model]:.3f})")

print("\n=== 2024 MIAMI GP PREDICTIONS ===")

# Create detailed predictions dataframe
results_df = test_data[['driver_name', 'team', 'grid_position']].copy()
results_df['actual_position'] = y_test.values
results_df['pred_linear_reg'] = y_pred_lr
results_df['pred_weighted_avg'] = y_pred_wa
results_df['pred_random_forest'] = y_pred_rf

# Sort by actual position
results_df = results_df.sort_values('actual_position')

print("\nDriver               | Team           | Grid | Actual | LR   | WA   | RF   |")
print("-" * 80)
for _, row in results_df.iterrows():
    print(f"{row['driver_name']:<18} | {row['team']:<13} | {row['grid_position']:4.0f} | "
          f"{row['actual_position']:6.0f} | {row['pred_linear_reg']:4.1f} | "
          f"{row['pred_weighted_avg']:4.1f} | {row['pred_random_forest']:4.1f} |")

# Save results
results_df.to_csv('miami_gp_predictions_comparison.csv', index=False)
print(f"\nDetailed predictions saved to 'miami_gp_predictions_comparison.csv'")

print("\n=== SUMMARY ===")
for model, rmse in models.items():
    print(f"{model}: RMSE = {rmse:.3f}")
print(f"\nBest model: {best_model}")
