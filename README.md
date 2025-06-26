# F1 Miami Grand Prix Interactive Dashboard

## Project Overview

This interactive dashboard aims to analyze F1 Miami Grand Prix prediction models and results. It combines data exploration, model comparison visualizations, and prediction accuracy plots into an intuitive user interface to offer rich visual insights into F1 predictions.

### Key Components:

1. **Setup and Data Loading**
   - Loads datasets: `miami_gp_predictions_comparison.csv` and `model_evaluation_summary.csv`.
   - Displays headers for a quick overview.

2. **Interactive Model Performance Comparison**
   - Compares models using metrics: RMSE, MAE, R² score, and accuracy.
   - Utilizes plotly to create a 2x2 grid of bar charts.

3. **Predicted vs Actual Positions Analysis**
   - Interactive scatter plot for actual vs predicted positions by model.
   - Includes hover information and error bands.

4. **2024 Miami GP Results Table**
   - Tabulates predictions with absolute errors.
   - Styles for podium and points finishers.

5. **Error Analysis Dashboard**
   - Placeholder for additional content focusing on error analysis.

### Notebook Files:
- **F1_Miami_Interactive_Dashboard.ipynb**: Enhanced interactive notebook located in the project directory.
- **Project_Notebook.ipynb**: Basic interactive notebook located in the `F1_Miami_General` directory.

## Prerequisites

To run the enhanced interactive dashboard, ensure you have the following installed:

- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `ipywidgets`
- Jupyter Notebook

## How to Run the Enhanced Interactive Dashboard

1. **Navigate to Project Directory:**
   ```bash
   cd ~/F1_Miami_General
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open the Notebook:**
   - Open `F1_Miami_Interactive_Dashboard.ipynb`.
   
4. **Run Notebook Cells:**
   - Sequentially run each cell to generate interactive visualizations.

5. **Install Required Libraries (if needed):**
   - If any libraries are missing, install them using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly ipywidgets
   ```

6. **Troubleshooting Tips:**
   - Ensure data files `miami_gp_predictions_comparison.csv` and `model_evaluation_summary.csv` are in the directory.
   - Check for any missing library errors and install accordingly.

## Driver Analysis Examples

- Analyze specific drivers using predefined functions, for example:
   ```python
   analyze_driver('Lando Norris')
   analyze_driver('Max Verstappen')
   analyze_driver('Charles Leclerc')
   ```

With these steps, you will be able to explore and interact with the F1 Miami Grand Prix prediction models effectively.
