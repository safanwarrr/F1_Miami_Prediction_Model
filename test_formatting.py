#!/usr/bin/env python3
"""
Test script to verify the data formatting in the notebook works correctly
"""

import pandas as pd

# Load the data
try:
    predictions = pd.read_csv('miami_gp_predictions_comparison.csv')
    eval_metrics = pd.read_csv('model_evaluation_summary.csv')
    print("✅ Data loaded successfully!")
    print(f"📊 Loaded {len(predictions)} driver predictions")
    print(f"📊 Loaded {len(eval_metrics)} model evaluations")
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
    print("Please ensure the CSV files are in the current directory")
    exit(1)

print("\n" + "="*60)
print("📋 Dataset Overview:")
print("="*60)

# Show original data first
print("\n🔍 Original predictions data (first 5 rows):")
print(predictions.head())

print("\n" + "-"*60)

# Create display-friendly version with integer positions
predictions_display = predictions.copy()
predictions_display['grid_position'] = predictions_display['grid_position'].fillna(0).astype(int)
predictions_display['actual_position'] = predictions_display['actual_position'].astype(int)

print("\n✨ Formatted predictions data (first 5 rows):")
print(predictions_display.head())

print("\n" + "-"*60)
print("\n📊 Model Evaluation Metrics:")
print(eval_metrics)

print("\n" + "="*60)
print("🎯 Formatting Test Results:")
print("="*60)

# Check the data types
print(f"\nOriginal grid_position dtype: {predictions['grid_position'].dtype}")
print(f"Formatted grid_position dtype: {predictions_display['grid_position'].dtype}")
print(f"Original actual_position dtype: {predictions['actual_position'].dtype}")
print(f"Formatted actual_position dtype: {predictions_display['actual_position'].dtype}")

# Show specific examples
print(f"\nExample values:")
print(f"Original grid_position[0]: {predictions['grid_position'].iloc[0]} (type: {type(predictions['grid_position'].iloc[0])})")
print(f"Formatted grid_position[0]: {predictions_display['grid_position'].iloc[0]} (type: {type(predictions_display['grid_position'].iloc[0])})")
print(f"Original actual_position[0]: {predictions['actual_position'].iloc[0]} (type: {type(predictions['actual_position'].iloc[0])})")
print(f"Formatted actual_position[0]: {predictions_display['actual_position'].iloc[0]} (type: {type(predictions_display['actual_position'].iloc[0])})")
