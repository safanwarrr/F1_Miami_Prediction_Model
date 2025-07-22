#!/usr/bin/env python3
"""
F1 Prediction Utilities
Helper functions for formatting predictions and handling position conversions
"""

import pandas as pd
import numpy as np

def convert_predictions_to_positions(predictions):
    """
    Convert float predictions to integer grid positions.
    
    Args:
        predictions: pandas Series or numpy array of float predictions
        
    Returns:
        pandas Series or numpy array of integer positions
    """
    if isinstance(predictions, pd.Series):
        return predictions.round().astype(int)
    else:
        return np.round(predictions).astype(int)

def convert_dataframe_predictions_to_integers(df, prediction_cols=['pred_weighted_avg', 'pred_linear_reg', 'pred_random_forest']):
    """
    Convert all model prediction columns in a DataFrame to integers.
    
    Args:
        df: pandas DataFrame with prediction columns
        prediction_cols: list of prediction column names to convert
        
    Returns:
        pandas DataFrame with integer predictions
    """
    df_copy = df.copy()
    
    for col in prediction_cols:
        if col in df_copy.columns:
            df_copy[col] = convert_predictions_to_positions(df_copy[col])
    
    return df_copy

def format_prediction_for_display(prediction_value):
    """
    Format a single prediction value for end-user display.
    
    Args:
        prediction_value: float prediction value
        
    Returns:
        string formatted as "P{position}" (e.g., "P3", "P15")
    """
    return f"P{round(prediction_value)}"

def create_display_friendly_dataframe(df, prediction_columns):
    """
    Create a display-friendly version of predictions dataframe with integer positions.
    
    Args:
        df: pandas DataFrame with prediction columns
        prediction_columns: list of column names containing float predictions
        
    Returns:
        pandas DataFrame with additional _display columns containing integer positions
    """
    display_df = df.copy()
    
    for col in prediction_columns:
        if col in df.columns:
            display_col_name = f"{col}_display"
            display_df[display_col_name] = convert_predictions_to_positions(df[col])
    
    return display_df

def format_prediction_table_output(df, driver_col='driver_name', team_col='team', 
                                 grid_col='grid_position', actual_col='actual_position',
                                 prediction_cols=['pred_weighted_avg', 'pred_linear_reg', 'pred_random_forest']):
    """
    Generate formatted table output for prediction results with integer positions.
    
    Args:
        df: pandas DataFrame with prediction results
        driver_col: column name for driver names
        team_col: column name for team names  
        grid_col: column name for grid positions
        actual_col: column name for actual positions
        prediction_cols: list of prediction column names
        
    Returns:
        formatted string table
    """
    output_lines = []
    
    # Header
    header = f"{'Driver':<18} | {'Team':<13} | {'Grid':<4} | {'Actual':<6}"
    for col in prediction_cols:
        # Extract model name from column (e.g., 'pred_weighted_avg' -> 'WA')
        if 'weighted' in col.lower():
            model_abbrev = 'WA'
        elif 'linear' in col.lower():
            model_abbrev = 'LR' 
        elif 'forest' in col.lower():
            model_abbrev = 'RF'
        else:
            model_abbrev = col[:4].upper()
        
        header += f" | {model_abbrev:<4}"
    
    output_lines.append(header)
    output_lines.append("-" * len(header))
    
    # Data rows
    for _, row in df.iterrows():
        line = f"{row[driver_col]:<18} | {row[team_col]:<13} | {row[grid_col]:<4.0f} | {row[actual_col]:<6.0f}"
        
        for col in prediction_cols:
            if col in row:
                line += f" | P{round(row[col]):<3.0f}"
            else:
                line += f" | {'N/A':<4}"
        
        output_lines.append(line)
    
    return "\n".join(output_lines)

def save_prediction_results(df, filename_base='miami_gp_predictions', 
                          prediction_cols=['pred_weighted_avg', 'pred_linear_reg', 'pred_random_forest']):
    """
    Save prediction results in both raw (float) and display-friendly (integer) formats.
    
    Args:
        df: pandas DataFrame with prediction results
        filename_base: base filename without extension
        prediction_cols: list of prediction column names to convert
        
    Returns:
        tuple of (raw_filename, display_filename)
    """
    # Save raw predictions with floats for model evaluation
    raw_filename = f"{filename_base}_comparison.csv"
    df.to_csv(raw_filename, index=False)
    
    # Create display version with integer positions
    display_df = create_display_friendly_dataframe(df, prediction_cols)
    display_filename = f"{filename_base}_display.csv" 
    display_df.to_csv(display_filename, index=False)
    
    return raw_filename, display_filename

def print_prediction_summary(df, prediction_cols=['pred_weighted_avg', 'pred_linear_reg', 'pred_random_forest'],
                           driver_col='driver_name', actual_col='actual_position'):
    """
    Print a summary of predictions vs actual results with integer positions.
    
    Args:
        df: pandas DataFrame with prediction results
        prediction_cols: list of prediction column names
        driver_col: column name for driver names
        actual_col: column name for actual positions
    """
    print("\n🏁 PREDICTION SUMMARY (Integer Positions)")
    print("=" * 60)
    
    for col in prediction_cols:
        if col in df.columns:
            # Model name formatting
            if 'weighted' in col.lower():
                model_name = 'Weighted Average'
            elif 'linear' in col.lower():
                model_name = 'Linear Regression'
            elif 'forest' in col.lower():
                model_name = 'Random Forest'
            else:
                model_name = col.replace('pred_', '').replace('_', ' ').title()
            
            print(f"\n{model_name} Predictions:")
            print("-" * 40)
            
            # Sort by predicted position and show top 10
            sorted_df = df.sort_values(col)
            
            for i, (_, row) in enumerate(sorted_df.head(10).iterrows(), 1):
                predicted_pos = round(row[col])
                actual_pos = int(row[actual_col])
                error = abs(predicted_pos - actual_pos)
                
                status = "✓" if error <= 2 else "✗" if error > 5 else "~"
                
                print(f"  P{i:2d}: {row[driver_col]:<18} "
                      f"(Predicted: P{predicted_pos:2d}, Actual: P{actual_pos:2d}) {status}")

# Example usage functions
def quick_position_check(prediction_value):
    """Quick check if a prediction value makes sense as a grid position."""
    if not isinstance(prediction_value, (int, float)):
        return False
    
    # F1 grids typically have 20 cars
    return 1 <= prediction_value <= 20

def validate_predictions_dataframe(df, prediction_cols):
    """
    Validate that prediction columns contain reasonable values.
    
    Args:
        df: pandas DataFrame
        prediction_cols: list of prediction column names
        
    Returns:
        dict with validation results
    """
    results = {}
    
    for col in prediction_cols:
        if col not in df.columns:
            results[col] = {'status': 'missing', 'message': f'Column {col} not found'}
            continue
            
        values = df[col].dropna()
        
        if len(values) == 0:
            results[col] = {'status': 'empty', 'message': f'Column {col} is empty'}
            continue
        
        min_val = values.min()
        max_val = values.max()
        
        if min_val < 1 or max_val > 25:  # Allow some margin beyond typical F1 grid
            results[col] = {
                'status': 'warning', 
                'message': f'Column {col} has unusual values (range: {min_val:.1f}-{max_val:.1f})'
            }
        else:
            results[col] = {
                'status': 'ok', 
                'message': f'Column {col} looks good (range: {min_val:.1f}-{max_val:.1f})'
            }
    
    return results

if __name__ == "__main__":
    # Example usage and testing
    print("F1 Prediction Utilities - Test Mode")
    print("="*50)
    
    # Test data
    test_predictions = pd.Series([1.2, 3.8, 5.1, 7.9, 12.3])
    print(f"Original predictions: {test_predictions.tolist()}")
    print(f"Integer positions: {convert_predictions_to_positions(test_predictions).tolist()}")
    
    # Test formatting
    for pred in test_predictions:
        print(f"{pred:.1f} -> {format_prediction_for_display(pred)}")
