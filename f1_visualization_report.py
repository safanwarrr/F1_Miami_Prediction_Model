#!/usr/bin/env python3
"""
F1 Miami Grand Prix Prediction - Visualization and Reporting
Creates comprehensive visualizations and generates final report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load all required data files"""
    try:
        predictions = pd.read_csv('miami_gp_predictions_comparison.csv')
        eval_metrics = pd.read_csv('model_evaluation_summary.csv')
        return predictions, eval_metrics
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def create_prediction_vs_actual_plot(predictions):
    """Create predicted vs actual positions scatter plot"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = ['pred_linear_reg', 'pred_weighted_avg', 'pred_random_forest']
    model_names = ['Linear Regression', 'Weighted Average', 'Random Forest']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (model_col, model_name, color) in enumerate(zip(models, model_names, colors)):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(predictions['actual_position'], predictions[model_col], 
                  alpha=0.7, s=100, color=color, edgecolors='white', linewidth=1)
        
        # Perfect prediction line
        ax.plot([1, 20], [1, 20], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')
        
        # Error bands
        x_line = np.linspace(1, 20, 100)
        ax.fill_between(x_line, x_line-2, x_line+2, alpha=0.2, color='gray', label='±2 Positions')
        ax.fill_between(x_line, x_line-1, x_line+1, alpha=0.3, color='gray', label='±1 Position')
        
        ax.set_xlabel('Actual Position', fontsize=12)
        ax.set_ylabel('Predicted Position', fontsize=12)
        ax.set_title(f'{model_name}\nPredicted vs Actual Positions', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 21)
        ax.set_ylim(0, 21)
        
        # Add R² score
        actual = predictions['actual_position']
        predicted = predictions[model_col]
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=11, fontweight='bold')
        
        if i == 0:
            ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('f1_predicted_vs_actual_positions.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_performance_comparison(eval_metrics):
    """Create model performance comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = eval_metrics['model']
    
    # RMSE comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, eval_metrics['rmse'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RMSE (positions)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars1, eval_metrics['rmse']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # MAE comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, eval_metrics['mae'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE (positions)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars2, eval_metrics['mae']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # R² comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, eval_metrics['r2'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_title('R² Score (Coefficient of Determination)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars3, eval_metrics['r2']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy within ±2 positions
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, eval_metrics['within_2_accuracy'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.set_title('Accuracy Within ±2 Positions', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars4, eval_metrics['within_2_accuracy']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('f1_model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_prediction_results_table(predictions):
    """Create a detailed results table visualization"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = predictions.sort_values('actual_position')[
        ['driver_name', 'team', 'grid_position', 'actual_position', 
         'pred_weighted_avg', 'pred_linear_reg', 'pred_random_forest']
    ].copy()
    
    # Calculate errors
    table_data['wa_error'] = abs(table_data['actual_position'] - table_data['pred_weighted_avg'])
    table_data['lr_error'] = abs(table_data['actual_position'] - table_data['pred_linear_reg'])
    table_data['rf_error'] = abs(table_data['actual_position'] - table_data['pred_random_forest'])
    
    # Round predictions
    table_data['pred_weighted_avg'] = table_data['pred_weighted_avg'].round(1)
    table_data['pred_linear_reg'] = table_data['pred_linear_reg'].round(1)
    table_data['pred_random_forest'] = table_data['pred_random_forest'].round(1)
    
    # Prepare final table
    display_data = []
    for _, row in table_data.iterrows():
        display_data.append([
            int(row['actual_position']),
            row['driver_name'],
            row['team'][:12],  # Truncate team names
            int(row['grid_position']),
            f"{row['pred_weighted_avg']:.1f}",
            f"{row['pred_linear_reg']:.1f}",
            f"{row['pred_random_forest']:.1f}",
            f"{row['wa_error']:.1f}",
            f"{row['lr_error']:.1f}",
            f"{row['rf_error']:.1f}"
        ])
    
    columns = ['Pos', 'Driver', 'Team', 'Grid', 'WA Pred', 'LR Pred', 'RF Pred', 'WA Err', 'LR Err', 'RF Err']
    
    # Create table
    table = ax.table(cellText=display_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code rows by position groups
    for i in range(1, len(display_data) + 1):
        pos = int(display_data[i-1][0])
        if pos <= 3:  # Podium
            color = '#FFD700'  # Gold
        elif pos <= 10:  # Points
            color = '#E6F3FF'  # Light blue
        elif pos <= 15:  # Midfield
            color = '#F0F0F0'  # Light gray
        else:  # Back of grid
            color = '#FFE6E6'  # Light red
        
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('2024 Miami Grand Prix - Detailed Prediction Results', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('f1_detailed_results_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_error_distribution_plot(predictions):
    """Create error distribution analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = ['pred_weighted_avg', 'pred_linear_reg', 'pred_random_forest']
    model_names = ['Weighted Average', 'Linear Regression', 'Random Forest']
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
    
    # Calculate errors
    errors = {}
    for model_col in models:
        errors[model_col] = predictions['actual_position'] - predictions[model_col]
    
    # Error distribution histogram
    ax1 = axes[0, 0]
    for model_col, model_name, color in zip(models, model_names, colors):
        ax1.hist(errors[model_col], bins=15, alpha=0.6, label=model_name, color=color)
    
    ax1.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Absolute error by position
    ax2 = axes[0, 1]
    positions = predictions['actual_position']
    wa_abs_errors = abs(errors['pred_weighted_avg'])
    
    ax2.scatter(positions, wa_abs_errors, alpha=0.7, s=100, color='#4ECDC4')
    ax2.set_xlabel('Actual Position', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Error vs Position (Weighted Average)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Position group accuracy
    ax3 = axes[1, 0]
    groups = ['Podium\n(P1-3)', 'Points\n(P4-10)', 'Midfield\n(P11-15)', 'Back\n(P16-20)']
    group_ranges = [(1, 3), (4, 10), (11, 15), (16, 20)]
    
    wa_group_rmse = []
    for start, end in group_ranges:
        mask = (predictions['actual_position'] >= start) & (predictions['actual_position'] <= end)
        if mask.any():
            group_errors = errors['pred_weighted_avg'][mask]
            rmse = np.sqrt(np.mean(group_errors ** 2))
            wa_group_rmse.append(rmse)
        else:
            wa_group_rmse.append(0)
    
    bars = ax3.bar(groups, wa_group_rmse, color=['#FFD700', '#4ECDC4', '#95A5A6', '#E74C3C'])
    ax3.set_ylabel('RMSE', fontsize=12)
    ax3.set_title('Prediction Accuracy by Position Group', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, wa_group_rmse):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Model comparison radar chart (simplified as bar chart)
    ax4 = axes[1, 1]
    metrics = ['RMSE', 'MAE', 'R²', '±2 Acc']
    wa_values = [3.93, 2.61, 0.536, 60.0]
    lr_values = [4.97, 3.49, 0.256, 50.0]
    rf_values = [4.91, 3.36, 0.275, 45.0]
    
    # Normalize values for comparison (lower is better for RMSE/MAE, higher for R²/Accuracy)
    norm_wa = [1/3.93, 1/2.61, 0.536, 60.0/100]
    norm_lr = [1/4.97, 1/3.49, 0.256, 50.0/100]
    norm_rf = [1/4.91, 1/3.36, 0.275, 45.0/100]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    ax4.bar(x - width, norm_wa, width, label='Weighted Average', color='#4ECDC4')
    ax4.bar(x, norm_lr, width, label='Linear Regression', color='#FF6B6B')
    ax4.bar(x + width, norm_rf, width, label='Random Forest', color='#45B7D1')
    
    ax4.set_xlabel('Metrics', fontsize=12)
    ax4.set_ylabel('Normalized Score', fontsize=12)
    ax4.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('f1_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_plot():
    """Create feature importance visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Weighted Average model weights (from our implementation)
    features = ['Historical Miami\nPerformance', 'Grid Position', 'Career Average\nPosition', 'Miami Experience']
    weights = [0.4, 0.3, 0.2, 0.1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95A5A6']
    
    bars = ax.bar(features, weights, color=colors)
    ax.set_ylabel('Feature Weight', fontsize=12)
    ax.set_title('Feature Importance in Weighted Average Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, weight in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{weight:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('f1_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_visualizations():
    """Generate all visualizations"""
    print("Loading data...")
    predictions, eval_metrics = load_data()
    
    if predictions is None or eval_metrics is None:
        print("Failed to load data. Please ensure all required CSV files are present.")
        return
    
    print("Creating visualizations...")
    
    print("1. Predicted vs Actual Positions Plot...")
    create_prediction_vs_actual_plot(predictions)
    
    print("2. Model Performance Comparison...")
    create_model_performance_comparison(eval_metrics)
    
    print("3. Detailed Results Table...")
    create_prediction_results_table(predictions)
    
    print("4. Error Distribution Analysis...")
    create_error_distribution_plot(predictions)
    
    print("5. Feature Importance Plot...")
    create_feature_importance_plot()
    
    print("\n✅ All visualizations created successfully!")
    print("Generated files:")
    print("• f1_predicted_vs_actual_positions.png")
    print("• f1_model_performance_comparison.png") 
    print("• f1_detailed_results_table.png")
    print("• f1_error_analysis.png")
    print("• f1_feature_importance.png")

if __name__ == "__main__":
    print("="*60)
    print("F1 MIAMI GP PREDICTION - VISUALIZATION GENERATOR")
    print("="*60)
    generate_visualizations()
