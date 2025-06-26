#!/usr/bin/env python3
"""
F1 Miami Grand Prix Data Collection Script
Collects race data from 2022, 2023, and 2024 Miami Grand Prix using FastF1
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Enable FastF1 cache for better performance
fastf1.Cache.enable_cache('fastf1_cache')

def collect_miami_race_data(year):
    """
    Collect race data for Miami Grand Prix for a specific year
    """
    print(f"\n=== Collecting {year} Miami Grand Prix Data ===")
    
    try:
        # Load the race session
        session = fastf1.get_session(year, 'Miami', 'R')
        session.load()
        
        # Get race results
        results = session.results
        print(f"Available columns: {list(results.columns)}")
        
        # Create comprehensive race data dictionary
        race_data = {
            'year': year,
            'race_name': 'Miami Grand Prix',
            'date': session.date,
            'total_laps': session.total_laps,
            'results': []
        }
        
        # Process each driver's results
        for idx, driver in results.iterrows():
            driver_data = {
                'position': driver['Position'],
                'classified_position': driver.get('ClassifiedPosition', driver['Position']),
                'driver_number': driver['DriverNumber'],
                'driver_code': driver['Abbreviation'],
                'driver_name': driver['FullName'],
                'first_name': driver['FirstName'],
                'last_name': driver['LastName'],
                'team': driver['TeamName'],
                'team_color': driver['TeamColor'],
                'grid_position': driver['GridPosition'],
                'points': driver['Points'],
                'time': str(driver['Time']) if pd.notna(driver['Time']) else 'DNF',
                'status': driver['Status'],
                'country_code': driver['CountryCode'],
                'q1_time': str(driver['Q1']) if pd.notna(driver['Q1']) else 'N/A',
                'q2_time': str(driver['Q2']) if pd.notna(driver['Q2']) else 'N/A',
                'q3_time': str(driver['Q3']) if pd.notna(driver['Q3']) else 'N/A'
            }
            race_data['results'].append(driver_data)
        
        return race_data
        
    except Exception as e:
        print(f"Error collecting {year} data: {str(e)}")
        return None

def analyze_driver_performance(all_race_data):
    """
    Analyze driver performance across all Miami races
    """
    print("\n=== Analyzing Driver Performance ===")
    
    driver_stats = {}
    
    for race_data in all_race_data:
        if race_data is None:
            continue
            
        year = race_data['year']
        for result in race_data['results']:
            driver_name = result['driver_name']
            
            if driver_name not in driver_stats:
                driver_stats[driver_name] = {
                    'races': [],
                    'positions': [],
                    'points': [],
                    'teams': set()
                }
            
            driver_stats[driver_name]['races'].append(year)
            driver_stats[driver_name]['positions'].append(result['position'] if result['position'] != 'NC' else 21)
            driver_stats[driver_name]['points'].append(result['points'])
            driver_stats[driver_name]['teams'].add(result['team'])
    
    # Calculate statistics
    performance_analysis = []
    for driver, stats in driver_stats.items():
        if len(stats['races']) > 0:
            avg_position = np.mean([p for p in stats['positions'] if isinstance(p, (int, float))])
            total_points = sum(stats['points'])
            avg_points = np.mean(stats['points'])
            
            performance_analysis.append({
                'driver': driver,
                'races_participated': len(stats['races']),
                'years': stats['races'],
                'teams': list(stats['teams']),
                'average_position': round(avg_position, 2),
                'total_points': total_points,
                'average_points': round(avg_points, 2),
                'positions': stats['positions']
            })
    
    # Sort by average position (best to worst)
    performance_analysis.sort(key=lambda x: x['average_position'])
    
    return performance_analysis

def analyze_team_performance(all_race_data):
    """
    Analyze team performance across all Miami races
    """
    print("\n=== Analyzing Team Performance ===")
    
    team_stats = {}
    
    for race_data in all_race_data:
        if race_data is None:
            continue
            
        year = race_data['year']
        for result in race_data['results']:
            team_name = result['team']
            
            if team_name not in team_stats:
                team_stats[team_name] = {
                    'races': [],
                    'positions': [],
                    'points': [],
                    'drivers': set()
                }
            
            team_stats[team_name]['races'].append(year)
            team_stats[team_name]['positions'].append(result['position'] if result['position'] != 'NC' else 21)
            team_stats[team_name]['points'].append(result['points'])
            team_stats[team_name]['drivers'].add(result['driver_name'])
    
    # Calculate team statistics
    team_analysis = []
    for team, stats in team_stats.items():
        if len(stats['races']) > 0:
            avg_position = np.mean([p for p in stats['positions'] if isinstance(p, (int, float))])
            total_points = sum(stats['points'])
            
            team_analysis.append({
                'team': team,
                'total_entries': len(stats['races']),
                'total_points': total_points,
                'average_position': round(avg_position, 2),
                'average_points_per_entry': round(total_points / len(stats['races']), 2),
                'drivers': list(stats['drivers'])
            })
    
    # Sort by total points (highest to lowest)
    team_analysis.sort(key=lambda x: x['total_points'], reverse=True)
    
    return team_analysis

def save_data_to_files(all_race_data, driver_performance, team_performance):
    """
    Save collected data to CSV files
    """
    print("\n=== Saving Data to Files ===")
    
    # Save raw race results
    all_results = []
    for race_data in all_race_data:
        if race_data is None:
            continue
        for result in race_data['results']:
            result['year'] = race_data['year']
            result['race_date'] = race_data['date']
            all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('miami_gp_race_results_2022_2024.csv', index=False)
    print("✓ Saved: miami_gp_race_results_2022_2024.csv")
    
    # Save driver performance analysis
    driver_df = pd.DataFrame(driver_performance)
    driver_df.to_csv('miami_gp_driver_performance_analysis.csv', index=False)
    print("✓ Saved: miami_gp_driver_performance_analysis.csv")
    
    # Save team performance analysis
    team_df = pd.DataFrame(team_performance)
    team_df.to_csv('miami_gp_team_performance_analysis.csv', index=False)
    print("✓ Saved: miami_gp_team_performance_analysis.csv")

def print_summary_statistics(all_race_data, driver_performance, team_performance):
    """
    Print summary statistics to console
    """
    print("\n" + "="*60)
    print("MIAMI GRAND PRIX DATA COLLECTION SUMMARY (2022-2024)")
    print("="*60)
    
    # Race summary
    valid_races = [r for r in all_race_data if r is not None]
    print(f"\nRaces collected: {len(valid_races)}")
    for race in valid_races:
        print(f"  • {race['year']}: {race['date'].strftime('%B %d, %Y')} ({len(race['results'])} drivers)")
    
    # Top performers
    print(f"\nTop 5 Drivers (by average position):")
    for i, driver in enumerate(driver_performance[:5], 1):
        print(f"  {i}. {driver['driver']} - Avg: P{driver['average_position']} ({driver['total_points']} pts)")
    
    print(f"\nTop 5 Teams (by total points):")
    for i, team in enumerate(team_performance[:5], 1):
        print(f"  {i}. {team['team']} - {team['total_points']} pts (Avg: P{team['average_position']})")
    
    # Winners by year
    print(f"\nMiami GP Winners:")
    for race in valid_races:
        winner = next((r for r in race['results'] if r['position'] == 1), None)
        if winner:
            print(f"  • {race['year']}: {winner['driver_name']} ({winner['team']})")

def main():
    """
    Main function to execute data collection and analysis
    """
    print("Starting F1 Miami Grand Prix Data Collection...")
    print("Using FastF1 library to collect race data from 2022, 2023, and 2024")
    
    # Years to collect data for
    years = [2022, 2023, 2024]
    
    # Collect race data for each year
    all_race_data = []
    for year in years:
        race_data = collect_miami_race_data(year)
        all_race_data.append(race_data)
    
    # Perform analysis
    driver_performance = analyze_driver_performance(all_race_data)
    team_performance = analyze_team_performance(all_race_data)
    
    # Save data to files
    save_data_to_files(all_race_data, driver_performance, team_performance)
    
    # Print summary
    print_summary_statistics(all_race_data, driver_performance, team_performance)
    
    print(f"\nData collection completed! Check the generated CSV files for detailed analysis.")

if __name__ == "__main__":
    main()
