#!/usr/bin/env python3
"""
F1 Prediction and Evaluation Script
Applies trained models to 2024 Miami GP and evaluates performance
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def calculate_prediction_metrics(actual, predicted, model_name):
    """Calculate comprehensive prediction metrics"""
    
    # Basic metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # R² score
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Position-specific accuracy
    exact_matches = np.sum(np.round(predicted) == actual)
    within_1_position = np.sum(np.abs(predicted - actual) <= 1)
    within_2_positions = np.sum(np.abs(predicted - actual) <= 2)
    within_3_positions = np.sum(np.abs(predicted - actual) <= 3)
    
    total_drivers = len(actual)
    
    metrics = {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'exact_accuracy': exact_matches / total_drivers * 100,
        'within_1_accuracy': within_1_position / total_drivers * 100,
        'within_2_accuracy': within_2_positions / total_drivers * 100,
        'within_3_accuracy': within_3_positions / total_drivers * 100
    }
    
    return metrics

def evaluate_top_n_predictions(actual, predicted, n=3):
    """Evaluate top-N prediction accuracy"""
    # Get actual top-N finishers
    actual_top_n = set(np.where(actual <= n)[0])
    
    # Get predicted top-N finishers
    predicted_top_n_indices = np.argsort(predicted)[:n]
    
    # Calculate intersection
    correct_predictions = len(actual_top_n.intersection(set(predicted_top_n_indices)))
    
    return correct_predictions, n, correct_predictions / n * 100

def evaluate_championship_points_accuracy(actual, predicted):
    """Evaluate how well we predict championship points allocation"""
    
    # F1 points system
    points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    
    def get_points(position):
        return points_system.get(int(position), 0)
    
    actual_points = [get_points(pos) for pos in actual]
    predicted_points = [get_points(pos) for pos in np.round(predicted)]
    
    points_mae = mean_absolute_error(actual_points, predicted_points)
    total_actual_points = sum(actual_points)
    total_predicted_points = sum(predicted_points)
    
    return {
        'actual_total_points': total_actual_points,
        'predicted_total_points': total_predicted_points,
        'points_mae': points_mae,
        'points_accuracy': (1 - abs(total_actual_points - total_predicted_points) / total_actual_points) * 100
    }

# Load the prediction results
print("="*80)
print("F1 MIAMI GRAND PRIX 2024 - PREDICTION AND EVALUATION")
print("="*80)

try:
    results = pd.read_csv('miami_gp_predictions_comparison.csv')
    print(f"✓ Loaded prediction results for {len(results)} drivers")
except FileNotFoundError:
    print("❌ Error: Prediction results file not found. Please run the model training script first.")
    exit(1)

print("\n=== 2024 MIAMI GP PREDICTED FINISHING ORDER ===")

# Define models and their prediction columns
models = {
    'Linear Regression': 'pred_linear_reg',
    'Weighted Average': 'pred_weighted_avg', 
    'Random Forest': 'pred_random_forest'
}

# Show predicted finishing order for each model
for model_name, pred_col in models.items():
    print(f"\n{model_name} - Predicted Finishing Order:")
    print("-" * 50)
    
    # Sort by predicted position
    model_predictions = results.sort_values(pred_col)
    
    print(f"{'Pos':<4} | {'Driver':<18} | {'Team':<13} | {'Predicted':<9}")
    print("-" * 50)
    
    for idx, (_, row) in enumerate(model_predictions.iterrows(), 1):
        print(f"{idx:<4} | {row['driver_name']:<18} | {row['team']:<13} | {row[pred_col]:<9.1f}")

print("\n" + "="*80)
print("ACTUAL 2024 MIAMI GP RESULTS")
print("="*80)

# Show actual results
actual_results = results.sort_values('actual_position')
print(f"{'Pos':<4} | {'Driver':<18} | {'Team':<13} | {'Grid':<5}")
print("-" * 45)

for _, row in actual_results.iterrows():
    print(f"{int(row['actual_position']):<4} | {row['driver_name']:<18} | {row['team']:<13} | {int(row['grid_position']):<5}")

print("\n" + "="*80)
print("COMPREHENSIVE EVALUATION METRICS")
print("="*80)

# Calculate metrics for each model
all_metrics = []
actual_positions = results['actual_position'].values

for model_name, pred_col in models.items():
    predicted_positions = results[pred_col].values
    metrics = calculate_prediction_metrics(actual_positions, predicted_positions, model_name)
    all_metrics.append(metrics)

# Display general metrics
print("\n=== GENERAL PREDICTION ACCURACY ===")
print(f"{'Model':<18} | {'MAE':<6} | {'RMSE':<6} | {'R²':<6} | {'Exact':<6} | {'±1':<6} | {'±2':<6} | {'±3':<6}")
print("-" * 85)

for metrics in all_metrics:
    print(f"{metrics['model']:<18} | {metrics['mae']:<6.2f} | {metrics['rmse']:<6.2f} | "
          f"{metrics['r2']:<6.3f} | {metrics['exact_accuracy']:<6.1f}% | "
          f"{metrics['within_1_accuracy']:<6.1f}% | {metrics['within_2_accuracy']:<6.1f}% | "
          f"{metrics['within_3_accuracy']:<6.1f}%")

# Top-N prediction accuracy
print("\n=== TOP-N PREDICTION ACCURACY ===")
for n in [1, 3, 5, 10]:
    print(f"\nTop-{n} Prediction Accuracy:")
    print(f"{'Model':<18} | {'Correct':<8} | {'Total':<6} | {'Accuracy':<8}")
    print("-" * 45)
    
    for model_name, pred_col in models.items():
        predicted_positions = results[pred_col].values
        correct, total, accuracy = evaluate_top_n_predictions(actual_positions, predicted_positions, n)
        print(f"{model_name:<18} | {correct:<8} | {total:<6} | {accuracy:<8.1f}%")

# Championship points evaluation
print("\n=== CHAMPIONSHIP POINTS ACCURACY ===")
print(f"{'Model':<18} | {'Actual Pts':<10} | {'Pred Pts':<9} | {'Points MAE':<11} | {'Points Acc':<11}")
print("-" * 75)

for model_name, pred_col in models.items():
    predicted_positions = results[pred_col].values
    points_metrics = evaluate_championship_points_accuracy(actual_positions, predicted_positions)
    
    print(f"{model_name:<18} | {points_metrics['actual_total_points']:<10.0f} | "
          f"{points_metrics['predicted_total_points']:<9.0f} | "
          f"{points_metrics['points_mae']:<11.2f} | {points_metrics['points_accuracy']:<11.1f}%")

# Detailed prediction analysis by position groups
print("\n=== PREDICTION ACCURACY BY POSITION GROUPS ===")

position_groups = [
    (1, 3, "Podium (P1-3)"),
    (4, 10, "Points (P4-10)"), 
    (11, 15, "Midfield (P11-15)"),
    (16, 20, "Back of Grid (P16-20)")
]

for model_name, pred_col in models.items():
    print(f"\n{model_name} - Accuracy by Position Group:")
    print(f"{'Group':<18} | {'Drivers':<8} | {'MAE':<6} | {'RMSE':<6}")
    print("-" * 45)
    
    predicted_positions = results[pred_col].values
    
    for start_pos, end_pos, group_name in position_groups:
        # Find drivers in this position group
        group_mask = (actual_positions >= start_pos) & (actual_positions <= end_pos)
        
        if np.any(group_mask):
            group_actual = actual_positions[group_mask]
            group_predicted = predicted_positions[group_mask]
            
            group_mae = mean_absolute_error(group_actual, group_predicted)
            group_rmse = np.sqrt(mean_squared_error(group_actual, group_predicted))
            
            print(f"{group_name:<18} | {len(group_actual):<8} | {group_mae:<6.2f} | {group_rmse:<6.2f}")

# Winner and podium prediction analysis
print("\n=== WINNER AND PODIUM PREDICTIONS ===")

actual_winner = results[results['actual_position'] == 1]['driver_name'].iloc[0]
actual_podium = results[results['actual_position'] <= 3]['driver_name'].tolist()

print(f"Actual Winner: {actual_winner}")
print(f"Actual Podium: {', '.join(actual_podium)}")
print()

for model_name, pred_col in models.items():
    # Predicted winner
    pred_winner_idx = results[pred_col].idxmin()
    pred_winner = results.loc[pred_winner_idx, 'driver_name']
    
    # Predicted podium
    pred_podium_indices = results.nsmallest(3, pred_col).index
    pred_podium = results.loc[pred_podium_indices, 'driver_name'].tolist()
    
    winner_correct = "✓" if pred_winner == actual_winner else "✗"
    podium_overlap = len(set(actual_podium).intersection(set(pred_podium)))
    
    print(f"{model_name}:")
    print(f"  Predicted Winner: {pred_winner} {winner_correct}")
    print(f"  Predicted Podium: {', '.join(pred_podium)}")
    print(f"  Podium Accuracy: {podium_overlap}/3 correct ({podium_overlap/3*100:.1f}%)")
    print()

# Biggest prediction errors
print("\n=== BIGGEST PREDICTION ERRORS ===")

for model_name, pred_col in models.items():
    print(f"\n{model_name} - Top 5 Biggest Errors:")
    print(f"{'Driver':<18} | {'Actual':<7} | {'Predicted':<10} | {'Error':<6}")
    print("-" * 50)
    
    results_copy = results.copy()
    results_copy['error'] = abs(results_copy['actual_position'] - results_copy[pred_col])
    biggest_errors = results_copy.nlargest(5, 'error')
    
    for _, row in biggest_errors.iterrows():
        print(f"{row['driver_name']:<18} | {row['actual_position']:<7.0f} | "
              f"{row[pred_col]:<10.1f} | {row['error']:<6.1f}")

# Model ranking and recommendation
print("\n" + "="*80)
print("FINAL MODEL EVALUATION SUMMARY")
print("="*80)

# Rank models by RMSE
model_ranking = sorted(all_metrics, key=lambda x: x['rmse'])

print("\nModel Performance Ranking (by RMSE):")
for i, metrics in enumerate(model_ranking, 1):
    print(f"{i}. {metrics['model']}: RMSE = {metrics['rmse']:.3f}, MAE = {metrics['mae']:.3f}")

best_model = model_ranking[0]
print(f"\n🏆 BEST PERFORMING MODEL: {best_model['model']}")
print(f"   RMSE: {best_model['rmse']:.3f} positions")
print(f"   MAE: {best_model['mae']:.3f} positions")
print(f"   Within ±2 positions accuracy: {best_model['within_2_accuracy']:.1f}%")

# Save detailed evaluation results
evaluation_summary = pd.DataFrame(all_metrics)
evaluation_summary.to_csv('model_evaluation_summary.csv', index=False)

print(f"\n📊 Detailed evaluation metrics saved to 'model_evaluation_summary.csv'")
print(f"📈 Prediction comparison data available in 'miami_gp_predictions_comparison.csv'")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
