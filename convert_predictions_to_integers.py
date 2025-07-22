#!/usr/bin/env python3
"""
Convert F1 Miami GP predictions from float to integer positions.
This script updates the CSV file with integer prediction values for all models.
"""

import pandas as pd
import numpy as np
from prediction_utils import convert_dataframe_predictions_to_integers

def main():
    # File paths
    input_file = "miami_gp_predictions_comparison.csv"
    output_file = "miami_gp_predictions_comparison.csv"
    backup_file = "miami_gp_predictions_comparison_backup.csv"
    
    print("🏁 Converting F1 Miami GP Predictions to Integer Positions")
    print("=" * 60)
    
    try:
        # Load the original data
        print(f"📁 Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} driver predictions")
        
        # Create backup
        print(f"💾 Creating backup at {backup_file}...")
        df.to_csv(backup_file, index=False)
        
        # Display original prediction ranges
        print("\n📊 Original Prediction Ranges (Float Values):")
        prediction_cols = ['pred_weighted_avg', 'pred_linear_reg', 'pred_random_forest']
        
        for col in prediction_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"  {col}: {min_val:.2f} to {max_val:.2f}")
        
        # Convert predictions to integers
        print(f"\n🔄 Converting predictions to integer positions...")
        df_integers = convert_dataframe_predictions_to_integers(df, prediction_cols)
        
        # Display new prediction ranges
        print("\n📊 New Prediction Ranges (Integer Values):")
        for col in prediction_cols:
            if col in df_integers.columns:
                min_val = df_integers[col].min()
                max_val = df_integers[col].max()
                print(f"  {col}: {min_val} to {max_val}")
        
        # Show sample conversions
        print("\n📋 Sample Conversions (First 5 Drivers):")
        print("-" * 80)
        for i in range(min(5, len(df))):
            driver = df.iloc[i]['driver_name']
            print(f"\n🏎️  {driver}:")
            for col in prediction_cols:
                if col in df.columns:
                    original = df.iloc[i][col]
                    converted = df_integers.iloc[i][col]
                    model_name = col.replace('pred_', '').replace('_', ' ').title()
                    print(f"    {model_name:18}: {original:6.2f} → P{converted:2d}")
        
        # Save the converted data
        print(f"\n💾 Saving integer predictions to {output_file}...")
        df_integers.to_csv(output_file, index=False)
        
        # Validation
        print(f"\n✅ Conversion completed successfully!")
        print(f"   - Original file backed up to: {backup_file}")
        print(f"   - Updated file saved as: {output_file}")
        print(f"   - All {len(prediction_cols)} model predictions converted to integers")
        
        # Show position distribution
        print("\n📈 Position Distribution Summary:")
        print("-" * 40)
        for col in prediction_cols:
            if col in df_integers.columns:
                model_name = col.replace('pred_', '').replace('_', ' ').title()
                positions = df_integers[col].value_counts().sort_index()
                unique_positions = len(positions)
                print(f"  {model_name:18}: {unique_positions} unique positions (P{positions.index.min()}-P{positions.index.max()})")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}")
        print("   Make sure you're running this script from the correct directory.")
        return 1
    
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
