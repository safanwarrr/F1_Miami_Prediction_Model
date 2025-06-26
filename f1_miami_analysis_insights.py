#!/usr/bin/env python3
"""
Enhanced F1 Miami Grand Prix Analysis Script
Provides detailed insights and visualizations from the collected data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the collected Miami GP data"""
    race_results = pd.read_csv('miami_gp_race_results_2022_2024.csv')
    driver_performance = pd.read_csv('miami_gp_driver_performance_analysis.csv')
    team_performance = pd.read_csv('miami_gp_team_performance_analysis.csv')
    
    return race_results, driver_performance, team_performance

def analyze_grid_vs_finish_performance(race_results):
    """Analyze how grid positions correlate with finishing positions"""
    print("\n=== GRID POSITION vs FINISHING POSITION ANALYSIS ===")
    
    # Remove DNFs for this analysis
    finished_races = race_results[race_results['status'] == 'Finished'].copy()
    
    # Calculate position gain/loss
    finished_races['position_change'] = finished_races['grid_position'] - finished_races['position']
    
    # Best overtakers (gained most positions)
    best_overtakers = finished_races.nlargest(10, 'position_change')
    print("\nTop 10 Overtaking Performances:")
    for idx, race in best_overtakers.iterrows():
        print(f"  {race['driver_name']} ({race['year']}): P{int(race['grid_position'])} → P{int(race['position'])} (+{int(race['position_change'])} positions)")
    
    # Biggest position losses
    biggest_losses = finished_races.nsmallest(5, 'position_change')
    print("\nBiggest Position Losses:")
    for idx, race in biggest_losses.iterrows():
        print(f"  {race['driver_name']} ({race['year']}): P{int(race['grid_position'])} → P{int(race['position'])} ({int(race['position_change'])} positions)")
    
    # Average position change by starting position
    grid_analysis = finished_races.groupby('grid_position')['position_change'].agg(['mean', 'count']).reset_index()
    grid_analysis = grid_analysis[grid_analysis['count'] >= 2]  # Only positions with 2+ samples
    
    print("\nAverage Position Change by Grid Position (min 2 samples):")
    for idx, row in grid_analysis.iterrows():
        direction = "gained" if row['mean'] > 0 else "lost"
        print(f"  P{int(row['grid_position'])}: {direction} {abs(row['mean']):.1f} positions on average")

def analyze_dnf_patterns(race_results):
    """Analyze DNF (Did Not Finish) patterns"""
    print("\n=== DNF ANALYSIS ===")
    
    dnfs = race_results[race_results['status'] != 'Finished'].copy()
    
    print(f"Total DNFs across 3 races: {len(dnfs)}")
    print(f"DNF rate: {len(dnfs)/len(race_results)*100:.1f}%")
    
    # DNFs by year
    dnf_by_year = dnfs.groupby('year').size()
    print("\nDNFs by year:")
    for year, count in dnf_by_year.items():
        print(f"  {year}: {count} DNFs")
    
    # Most common DNF reasons
    dnf_reasons = dnfs['status'].value_counts()
    print("\nMost common DNF reasons:")
    for reason, count in dnf_reasons.items():
        print(f"  {reason}: {count}")
    
    # Drivers with most DNFs
    driver_dnfs = dnfs['driver_name'].value_counts()
    print("\nDrivers with most DNFs:")
    for driver, count in driver_dnfs.head(5).items():
        print(f"  {driver}: {count}")

def analyze_points_efficiency(race_results, driver_performance):
    """Analyze points scoring efficiency"""
    print("\n=== POINTS EFFICIENCY ANALYSIS ===")
    
    # Points per race for drivers with multiple participations
    multi_race_drivers = driver_performance[driver_performance['races_participated'] > 1].copy()
    
    print("Points efficiency (drivers with 2+ races):")
    efficiency_ranking = multi_race_drivers.sort_values('average_points', ascending=False)
    
    for idx, driver in efficiency_ranking.head(10).iterrows():
        print(f"  {driver['driver']}: {driver['average_points']:.1f} pts/race ({driver['total_points']:.0f} total)")

def analyze_team_consistency(race_results):
    """Analyze team consistency across races"""
    print("\n=== TEAM CONSISTENCY ANALYSIS ===")
    
    # Calculate standard deviation of positions for each team
    finished_races = race_results[race_results['status'] == 'Finished'].copy()
    
    team_consistency = finished_races.groupby('team')['position'].agg(['mean', 'std', 'count']).reset_index()
    team_consistency = team_consistency[team_consistency['count'] >= 4]  # Teams with 4+ finishes
    team_consistency = team_consistency.sort_values('std')
    
    print("Most consistent teams (by position std dev, min 4 finishes):")
    for idx, team in team_consistency.head(8).iterrows():
        print(f"  {team['team']}: {team['std']:.1f} std dev (avg P{team['mean']:.1f})")

def analyze_qualifying_vs_race_performance(race_results):
    """Analyze relationship between qualifying and race performance"""
    print("\n=== QUALIFYING vs RACE PERFORMANCE ===")
    
    # Focus on races where we have qualifying data (grid position)
    valid_grid = race_results[race_results['grid_position'] > 0].copy()
    finished_races = valid_grid[valid_grid['status'] == 'Finished'].copy()
    
    # Correlation between grid and finish position
    correlation = finished_races['grid_position'].corr(finished_races['position'])
    print(f"Grid-to-finish correlation: {correlation:.3f}")
    
    # Drivers who consistently outperform their grid position
    finished_races['outperformance'] = finished_races['grid_position'] - finished_races['position']
    
    driver_outperformance = finished_races.groupby('driver_name')['outperformance'].agg(['mean', 'count']).reset_index()
    driver_outperformance = driver_outperformance[driver_outperformance['count'] >= 2]
    best_race_day_drivers = driver_outperformance.sort_values('mean', ascending=False)
    
    print("\nBest race day performers (vs qualifying, min 2 races):")
    for idx, driver in best_race_day_drivers.head(8).iterrows():
        direction = "gained" if driver['mean'] > 0 else "lost"
        print(f"  {driver['driver_name']}: {direction} {abs(driver['mean']):.1f} positions on average")

def analyze_championship_impact(race_results):
    """Analyze Miami GP's impact on championship standings"""
    print("\n=== CHAMPIONSHIP IMPACT ANALYSIS ===")
    
    # Points scored by year
    points_by_year = race_results.groupby(['year', 'driver_name'])['points'].sum().reset_index()
    
    for year in [2022, 2023, 2024]:
        year_points = points_by_year[points_by_year['year'] == year].sort_values('points', ascending=False)
        print(f"\n{year} Miami GP Championship Points:")
        for idx, driver in year_points.head(10).iterrows():
            if driver['points'] > 0:
                print(f"  {driver['driver_name']}: {driver['points']:.0f} pts")

def generate_race_summaries(race_results):
    """Generate detailed summaries for each race"""
    print("\n=== RACE SUMMARIES ===")
    
    for year in [2022, 2023, 2024]:
        year_data = race_results[race_results['year'] == year].sort_values('position')
        
        print(f"\n{year} MIAMI GRAND PRIX SUMMARY:")
        print(f"Date: {year_data.iloc[0]['race_date']}")
        
        # Podium
        podium = year_data.head(3)
        print("Podium:")
        positions = ['Winner', '2nd', '3rd']
        for i, (idx, driver) in enumerate(podium.iterrows()):
            print(f"  {positions[i]}: {driver['driver_name']} ({driver['team']})")
        
        # Pole position (lowest grid position among finishers)
        pole_sitter = year_data[year_data['grid_position'] == year_data['grid_position'].min()].iloc[0]
        print(f"Pole Position: {pole_sitter['driver_name']} ({pole_sitter['team']})")
        
        # DNFs
        dnfs = year_data[year_data['status'] != 'Finished']
        if len(dnfs) > 0:
            print("DNFs:")
            for idx, driver in dnfs.iterrows():
                print(f"  {driver['driver_name']}: {driver['status']}")
        
        # Notable performances
        finished = year_data[year_data['status'] == 'Finished']
        if len(finished) > 0:
            finished['position_change'] = finished['grid_position'] - finished['position']
            best_overtaker = finished.loc[finished['position_change'].idxmax()]
            if best_overtaker['position_change'] > 0:
                print(f"Best Overtaker: {best_overtaker['driver_name']} (P{int(best_overtaker['grid_position'])} → P{int(best_overtaker['position'])}, +{int(best_overtaker['position_change'])})")

def main():
    """Main analysis function"""
    print("ENHANCED F1 MIAMI GRAND PRIX ANALYSIS (2022-2024)")
    print("=" * 60)
    
    try:
        # Load data
        race_results, driver_performance, team_performance = load_data()
        
        # Run analyses
        analyze_grid_vs_finish_performance(race_results)
        analyze_dnf_patterns(race_results)
        analyze_points_efficiency(race_results, driver_performance)
        analyze_team_consistency(race_results)
        analyze_qualifying_vs_race_performance(race_results)
        analyze_championship_impact(race_results)
        generate_race_summaries(race_results)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please run the data collection script first.")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
