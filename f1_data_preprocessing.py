#!/usr/bin/env python3
"""
F1 Data Preprocessing Script
Handles data cleaning and feature engineering for Miami GP datasets
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the race results data
race_results = pd.read_csv('miami_gp_race_results_2022_2024.csv')

# Data Cleaning
print("=== DATA CLEANING ===")

# Handle missing values and inconsistencies
print("Handling missing values...")
race_results.fillna({'status': 'Finished', 'points': 0, 'grid_position': race_results['position']}, inplace=True)

# Standardize team and driver names
print("Standardizing team and driver names...")
name_replacements = {
    'Alfa Romeo': 'Alfa Romeo Racing',
    'Red Bull Racing': 'Red Bull',
    'Haas F1 Team': 'Haas',
    'Kick Sauber': 'Sauber'
}
race_results['team'] = race_results['team'].replace(name_replacements)

# Feature Engineering
print("=== FEATURE ENGINEERING ===")

# Calculate average finishing positions for each driver in 2024 races prior to Miami
print("Calculating average finishing positions for 2024...")
race_results_2024 = race_results[race_results['year'] == 2024].copy()
race_results_2024['is_miami'] = race_results_2024['race_date'].apply(lambda x: 'Miami' in str(x))
avg_finishing_positions_2024 = race_results_2024[race_results_2024['is_miami'] == False].groupby('driver_name')['position'].mean().reset_index()
avg_finishing_positions_2024.rename(columns={'position': 'avg_position_2024_pre_miami'}, inplace=True)

# Calculate average points per race for each driver
print("Calculating average points per race...")
avg_points = race_results.groupby('driver_name')['points'].mean().reset_index()
avg_points.rename(columns={'points': 'avg_points_per_race'}, inplace=True)

# Merge average finishing positions and points data into race results
race_results = race_results.merge(avg_finishing_positions_2024, on='driver_name', how='left')
race_results = race_results.merge(avg_points, on='driver_name', how='left')

# Extract performance trends (e.g., recent form, improvement over previous races)
print("Extracting performance trends...")
race_results.sort_values(by=['driver_name', 'race_date'], inplace=True)
race_results['performance_trend'] = race_results.groupby('driver_name')['position'].diff().fillna(0).apply(lambda x: 'Improved' if x < 0 else ('Declined' if x > 0 else 'Same'))

# Incorporate historical Miami performance for each driver (if available)
print("Incorporating historical Miami performance...")
miami_performance = race_results[race_results['race_date'].apply(lambda x: 'Miami' in str(x))].groupby('driver_name')['position'].mean().reset_index()
miami_performance.rename(columns={'position': 'avg_miami_position'}, inplace=True)
race_results = race_results.merge(miami_performance, on='driver_name', how='left')

# Save the preprocessed data
print("Saving preprocessed data...")
race_results.to_csv('preprocessed_miami_gp_data.csv', index=False)
print("Preprocessed data saved as 'preprocessed_miami_gp_data.csv'")

print("Data preprocessing complete!")
