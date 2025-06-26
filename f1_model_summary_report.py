#!/usr/bin/env python3
"""
F1 Model Training Summary Report
Provides comprehensive analysis of model performance and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load prediction results
results = pd.read_csv('miami_gp_predictions_comparison.csv')

print("="*80)
print("F1 MIAMI GRAND PRIX 2024 - MODEL TRAINING SUMMARY REPORT")
print("="*80)

print("\n=== EXECUTIVE SUMMARY ===")
print("Three models were trained to predict 2024 Miami GP finishing positions:")
print("1. Linear Regression - Statistical model using historical performance features")
print("2. Weighted Average - Domain-specific model with expert-weighted features")
print("3. Random Forest - Ensemble model for comparison")

print("\n=== TRAINING APPROACH ===")
print("• Training Data: 2022-2023 Miami GP races (40 entries)")
print("• Test Data: 2024 Miami GP race (20 entries)")
print("• Features Used:")
print("  - Grid position (qualifying result)")
print("  - Historical Miami average position (2022-2023)")
print("  - Career average position (2022-2023)")
print("  - Career average points (2022-2023)")
print("  - Miami race experience count")

print("\n=== MODEL PERFORMANCE COMPARISON ===")

# Calculate metrics for each model
models = ['Linear Regression', 'Weighted Average', 'Random Forest']
pred_cols = ['pred_linear_reg', 'pred_weighted_avg', 'pred_random_forest']

metrics = {}
for model, pred_col in zip(models, pred_cols):
    rmse = np.sqrt(np.mean((results['actual_position'] - results[pred_col])**2))
    mae = np.mean(np.abs(results['actual_position'] - results[pred_col]))
    
    # Calculate R²
    ss_res = np.sum((results['actual_position'] - results[pred_col]) ** 2)
    ss_tot = np.sum((results['actual_position'] - np.mean(results['actual_position'])) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics[model] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}

print(f"{'Model':<18} | {'RMSE':<6} | {'MAE':<6} | {'R²':<6}")
print("-" * 45)
for model, metric in metrics.items():
    print(f"{model:<18} | {metric['RMSE']:<6.3f} | {metric['MAE']:<6.3f} | {metric['R²']:<6.3f}")

# Find best model
best_model = min(metrics.keys(), key=lambda k: metrics[k]['RMSE'])
print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
print(f"   RMSE: {metrics[best_model]['RMSE']:.3f} positions")

print("\n=== DETAILED PREDICTION ANALYSIS ===")

# Calculate prediction accuracy by position ranges
def analyze_predictions_by_range(results, pred_col, model_name):
    """Analyze prediction accuracy by finishing position ranges"""
    print(f"\n{model_name} - Accuracy by Position Range:")
    
    ranges = [(1, 5, "Podium/Top 5"), (6, 10, "Points Positions"), 
              (11, 15, "Midfield"), (16, 20, "Back of Grid")]
    
    for start, end, label in ranges:
        range_data = results[(results['actual_position'] >= start) & 
                           (results['actual_position'] <= end)]
        if len(range_data) > 0:
            range_rmse = np.sqrt(np.mean((range_data['actual_position'] - range_data[pred_col])**2))
            print(f"  {label:<15}: {len(range_data)} drivers, RMSE = {range_rmse:.3f}")

# Analyze best model in detail
best_pred_col = pred_cols[models.index(best_model)]
analyze_predictions_by_range(results, best_pred_col, best_model)

print("\n=== NOTABLE PREDICTIONS ===")

# Best predictions (smallest errors)
results['error'] = abs(results['actual_position'] - results[best_pred_col])
best_predictions = results.nsmallest(5, 'error')
worst_predictions = results.nlargest(5, 'error')

print(f"\nBest Predictions ({best_model}):")
print(f"{'Driver':<18} | {'Team':<12} | {'Actual':<6} | {'Predicted':<9} | {'Error':<5}")
print("-" * 60)
for _, row in best_predictions.iterrows():
    pred_val = row[best_pred_col]
    print(f"{row['driver_name']:<18} | {row['team']:<12} | {row['actual_position']:<6.0f} | "
          f"{pred_val:<9.1f} | {row['error']:<5.1f}")

print(f"\nWorst Predictions ({best_model}):")
print(f"{'Driver':<18} | {'Team':<12} | {'Actual':<6} | {'Predicted':<9} | {'Error':<5}")
print("-" * 60)
for _, row in worst_predictions.iterrows():
    pred_val = row[best_pred_col]
    print(f"{row['driver_name']:<18} | {row['team']:<12} | {row['actual_position']:<6.0f} | "
          f"{pred_val:<9.1f} | {row['error']:<5.1f}")

print("\n=== KEY INSIGHTS ===")

# Winner prediction analysis
winner_actual = results[results['actual_position'] == 1]['driver_name'].iloc[0]
winner_predictions = {}
for model, pred_col in zip(models, pred_cols):
    pred_winner_idx = results[pred_col].idxmin()
    pred_winner = results.loc[pred_winner_idx, 'driver_name']
    pred_position = results.loc[pred_winner_idx, pred_col]
    winner_predictions[model] = (pred_winner, pred_position)

print(f"Race Winner: {winner_actual}")
print("Winner Predictions:")
for model, (pred_winner, pred_pos) in winner_predictions.items():
    correct = "✓" if pred_winner == winner_actual else "✗"
    print(f"  {model}: {pred_winner} (P{pred_pos:.1f}) {correct}")

# Podium prediction analysis
actual_podium = set(results[results['actual_position'] <= 3]['driver_name'])
print(f"\nActual Podium: {', '.join(sorted(actual_podium))}")

for model, pred_col in zip(models, pred_cols):
    pred_podium = set(results.nsmallest(3, pred_col)['driver_name'])
    podium_accuracy = len(actual_podium.intersection(pred_podium)) / 3
    print(f"{model} Podium Accuracy: {podium_accuracy:.1%} "
          f"({len(actual_podium.intersection(pred_podium))}/3 correct)")

print("\n=== MODEL INTERPRETATION ===")

if best_model == "Weighted Average":
    print("The Weighted Average model performed best, suggesting that:")
    print("• Domain expertise in feature weighting is valuable")
    print("• Historical Miami performance (40% weight) is most predictive")
    print("• Grid position (30% weight) remains important")
    print("• Career average (20% weight) provides baseline performance")
    print("• Miami experience (10% weight) offers slight advantage")

elif best_model == "Linear Regression":
    print("The Linear Regression model performed best, indicating that:")
    print("• Linear relationships exist between features and finishing position")
    print("• Statistical modeling can capture F1 performance patterns")
    print("• Feature combination through learned coefficients is effective")

else:  # Random Forest
    print("The Random Forest model performed best, suggesting that:")
    print("• Non-linear relationships exist in the data")
    print("• Feature interactions are important for prediction")
    print("• Ensemble methods can capture complex patterns")

print("\n=== LIMITATIONS AND RECOMMENDATIONS ===")
print("Limitations:")
print("• Limited training data (only 40 samples from 2 years)")
print("• F1 is highly dynamic with rule changes and car development")
print("• External factors (weather, incidents) not captured")
print("• Driver/team changes between seasons not fully modeled")

print("\nRecommendations for Improvement:")
print("• Incorporate more historical data from other circuits")
print("• Add weather and track condition features")
print("• Include car development/upgrade information")
print("• Model driver-team combination effects")
print("• Use ensemble methods combining multiple model types")

print("\n=== BUSINESS VALUE ===")
print("These models can provide value for:")
print("• Sports betting and prediction markets")
print("• Fantasy F1 team selection")
print("• Media and broadcast insights")
print("• Team strategy planning")
print("• Fan engagement applications")

avg_error = np.mean(results['error'])
print(f"\nAverage prediction error: {avg_error:.1f} positions")
print(f"This represents approximately {avg_error/20*100:.1f}% accuracy in position prediction")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save summary statistics
summary_stats = pd.DataFrame(metrics).T
summary_stats.to_csv('model_performance_summary.csv')
print(f"\nSummary statistics saved to 'model_performance_summary.csv'")

if __name__ == "__main__":
    print("\nTo run this report: python3 f1_model_summary_report.py")
