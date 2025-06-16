import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy import stats
import os

# Create charts directory if it doesn't exist
if not os.path.exists('charts'):
    os.makedirs('charts')

# Set up visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'

def connect_to_db():
    """Establish connection to the simulation database"""
    try:
        conn = sqlite3.connect("cartel_simulation.db")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def get_table_as_df(table_name):
    """Retrieve a database table as a pandas DataFrame"""
    conn = connect_to_db()
    if conn is None:
        return pd.DataFrame()
    
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        return df
    except pd.errors.DatabaseError as e:
        print(f"Error reading table {table_name}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def analyze_production():
    """Analyze production events data with enhanced visualizations"""
    print("\n=== Analyzing Production Events ===")
    df = get_table_as_df("ProductionEvents")
    
    if df.empty:
        print("No production data available")
        return pd.DataFrame()
    
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Summary stats
    print(f"\nTotal production events: {len(df)}")
    print(f"Time period: {df.index.min()} to {df.index.max()}")
    print(f"Duration: {df.index.max() - df.index.min()}")
    
    # Success vs failure rates
    success_df = df[df['event_type'] == 'Harvest Success']
    failure_df = df[df['event_type'] == 'Crop Failure']
    
    success_rate = len(success_df)/len(df)
    failure_rate = len(failure_df)/len(df)
    
    print(f"\nSuccess rate: {success_rate:.1%}")
    print(f"Failure rate: {failure_rate:.1%}")
    
    # Production by farmer - enhanced visualization
    if not success_df.empty:
        farmer_prod = success_df.groupby('location')['quantity'].sum().sort_values(ascending=False)
        print("\nTop producers by quantity:")
        print(farmer_prod.head(10))
        
        # Enhanced visualization with distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot of top producers
        farmer_prod.head(10).plot(kind='bar', ax=ax1, title='Top 10 Producers by Quantity')
        ax1.set_ylabel('Total Quantity Produced')
        ax1.set_xlabel('Farmer Location')
        ax1.tick_params(axis='x', rotation=45)
        
        # Distribution of production quantities
        sns.histplot(success_df['quantity'], bins=20, kde=True, ax=ax2)
        ax2.set_title('Distribution of Production Quantities')
        ax2.set_xlabel('Quantity per Event')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('charts/farmer_performance_enhanced.png', bbox_inches='tight')
        plt.close()
    
    # Time series analysis with trend line
    if not success_df.empty:
        hourly_prod = success_df['quantity'].resample('5min').sum()
        
        plt.figure(figsize=(14, 7))
        ax = hourly_prod.plot(title='Production Volume Over Time (5-minute intervals)', marker='o', markersize=5)
        
        # Add trend line only if we have enough data points and variation
        if len(hourly_prod) > 2 and not (hourly_prod == hourly_prod.iloc[0]).all():
            try:
                x = np.arange(len(hourly_prod))
                y = hourly_prod.values
                trend = np.polyfit(x, y, 1)
                ax.plot(hourly_prod.index, np.poly1d(trend)(x), 'r--', 
                       label=f'Trend (Slope: {trend[0]:.2f})')
                ax.legend()
            except Exception as e:
                print(f"Could not calculate trend line: {e}")
        
        plt.ylabel('Quantity Produced')
        plt.xlabel('Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('charts/production_over_time_enhanced.png')
        plt.close()
    
    return df

def analyze_processing():
    """Analyze processing events data with enhanced visualizations"""
    print("\n=== Analyzing Processing Events ===")
    df = get_table_as_df("ProcessingEvents")
    
    if df.empty:
        print("No processing data available")
        return pd.DataFrame()
    
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    
    # Summary stats
    print(f"\nTotal processing events: {len(df)}")
    
    # Enhanced success analysis
    success_rate = df['success'].mean()
    print(f"\nOverall success rate: {success_rate:.1%}")
    
    # Success by event type with confidence intervals
    success_by_type = df.groupby('event_type')['success'].agg(['mean', 'count', 'std'])
    success_by_type['ci'] = 1.96 * success_by_type['std'] / np.sqrt(success_by_type['count'])
    print("\nSuccess rate by event type with confidence intervals:")
    print(success_by_type)
    
    # Enhanced event frequency visualization
    event_freq = df['event_type'].value_counts()
    print("\nEvent frequency:")
    print(event_freq)
    
    # Time between events analysis
    df['time_diff'] = df.index.to_series().diff().dt.total_seconds()
    
    # Enhanced visualization for quality assessments
    if 'Quality Assessment' in df['event_type'].values:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Time differences by outcome
        time_diff_data = df[df['event_type'] == 'Quality Assessment']
        sns.boxplot(x='success', y='time_diff', data=time_diff_data, ax=ax1)
        ax1.set_title('Time Between Quality Assessments by Outcome')
        ax1.set_ylabel('Seconds between assessments')
        ax1.set_xlabel('Outcome')
        ax1.set_xticklabels(['Failure', 'Success'])
        
        # Distribution of time differences
        sns.histplot(time_diff_data['time_diff'], bins=30, kde=True, ax=ax2)
        ax2.set_title('Distribution of Time Between Assessments')
        ax2.set_xlabel('Seconds between assessments')
        
        plt.tight_layout()
        plt.savefig('charts/quality_assessment_times_enhanced.png')
        plt.close()
    
    # Enhanced success rate over time visualization
    if len(df) > 10:  # Only plot if we have enough data points
        try:
            rolling_success = df['success'].rolling('5min').mean()
            
            plt.figure(figsize=(14, 7))
            rolling_success.plot(title='5-Minute Rolling Average of Processing Success Rate', 
                               linewidth=2)
            
            # Add horizontal line at overall success rate
            plt.axhline(y=success_rate, color='r', linestyle='--', 
                       label=f'Overall Rate: {success_rate:.1%}')
            
            plt.ylabel('Success Rate')
            plt.xlabel('Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('charts/processing_success_rolling_enhanced.png')
            plt.close()
        except Exception as e:
            print(f"\nCould not calculate rolling success rate: {e}")
    
    return df

def analyze_morale():
    """Analyze morale events data with enhanced visualizations"""
    print("\n=== Analyzing Morale Events ===")
    df = get_table_as_df("MoraleEvents")
    
    if df.empty:
        print("No morale data available")
        return pd.DataFrame()
    
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    
    # Summary stats
    print(f"\nTotal morale updates: {len(df)}")
    print(f"Final morale: {df['current_morale'].iloc[-1]:.2f}")
    print(f"Morale range: {df['current_morale'].min():.2f} to {df['current_morale'].max():.2f}")
    
    # Enhanced impact analysis
    impactful = df[df['impact'] != 0]
    print("\nImpactful events:")
    print(impactful[['event_type', 'impact', 'current_morale']].describe())
    
    # Enhanced visualization
    plt.figure(figsize=(14, 7))
    ax = df['current_morale'].plot(title='Morale Over Time with Impact Events', 
                                  linewidth=2, marker='o', markersize=5)
    
    # Mark impactful events with improved annotations
    for idx, row in impactful.iterrows():
        plt.axvline(x=idx, color='red', alpha=0.3, linestyle='--')
        plt.text(idx, row['current_morale'], 
                f"{row['event_type']}\nΔ{row['impact']:.2f}", 
                ha='right', va='center', rotation=90, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.ylabel('Morale Level')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/morale_over_time_enhanced.png')
    plt.close()
    
    # Enhanced impact distribution analysis
    if not impactful.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram with KDE
        sns.histplot(impactful['impact'], bins=15, kde=True, ax=ax1)
        ax1.set_title('Distribution of Morale Impacts')
        ax1.set_xlabel('Impact Value')
        
        # Boxplot by event type
        if len(impactful['event_type'].unique()) > 1:
            sns.boxplot(x='event_type', y='impact', data=impactful, ax=ax2)
            ax2.set_title('Impact Distribution by Event Type')
            ax2.set_ylabel('Impact Value')
            ax2.set_xlabel('Event Type')
            ax2.tick_params(axis='x', rotation=45)
        else:
            fig.delaxes(ax2)  # Remove second plot if not enough categories
        
        plt.tight_layout()
        plt.savefig('charts/morale_impact_distribution_enhanced.png')
        plt.close()
    
    return df

def analyze_sabotage():
    """Analyze sabotage events data with enhanced visualizations"""
    print("\n=== Analyzing Sabotage Events ===")
    df = get_table_as_df("SabotageEvents")
    
    if df.empty:
        print("No sabotage events recorded")
        return None
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Enhanced summary stats
    print(f"\nTotal sabotage events: {len(df)}")
    success_rate = df['success'].mean()
    print(f"Success rate: {success_rate:.1%}")
    
    # Enhanced damage analysis
    print("\nDamage statistics by sabotage type:")
    damage_stats = df.groupby('sabotage_type')['damage_estimate'].agg(['mean', 'median', 'std', 'count'])
    print(damage_stats)
    
    # Target analysis with proportions
    print("\nTarget frequency:")
    target_counts = df['target_type'].value_counts(normalize=True)
    print(target_counts)
    
    # Enhanced visualization
    plt.figure(figsize=(14, 7))
    
    # Create subplots
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    
    # Main damage plot
    sns.barplot(x='sabotage_type', y='damage_estimate', hue='success', data=df,
               estimator=np.mean, ci=95, ax=ax1)
    ax1.set_title('Average Damage by Sabotage Type and Success (with 95% CI)')
    ax1.set_ylabel('Average Damage Estimate')
    ax1.set_xlabel('Sabotage Type')
    ax1.legend(title='Success')
    
    # Target distribution
    df['target_type'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                        ax=ax2, title='Target Distribution')
    ax2.set_ylabel('')
    
    # Damage distribution
    sns.histplot(df['damage_estimate'], bins=20, kde=True, ax=ax3)
    ax3.set_title('Damage Estimate Distribution')
    ax3.set_xlabel('Damage Estimate')
    
    plt.tight_layout()
    plt.savefig('charts/sabotage_analysis_enhanced.png')
    plt.close()
    
    return df

def analyze_financials():
    """Analyze financial aspects with enhanced visualizations"""
    print("\n=== Analyzing Financial Data ===")
    # Get simulation run data
    sim_df = get_table_as_df("SimulationRuns")
    
    if sim_df.empty:
        print("No simulation run data available")
        return pd.DataFrame(), None
    
    # Enhanced financial summary
    print("\nSimulation Financial Summary:")
    financial_summary = sim_df[['start_time', 'end_time', 'final_funds', 'status']].copy()
    financial_summary['duration'] = pd.to_datetime(financial_summary['end_time']) - pd.to_datetime(financial_summary['start_time'])
    print(financial_summary)
    
    # Get payroll data from ProcessingEvents
    conn = connect_to_db()
    payroll_df = pd.read_sql(
        "SELECT * FROM ProcessingEvents WHERE event_type = 'Payroll'", 
        conn)
    conn.close()
    
    if payroll_df.empty:
        print("\nNo payroll events recorded")
        return sim_df, None
    
    payroll_df['timestamp'] = pd.to_datetime(payroll_df['timestamp'])
    
    # Enhanced payroll analysis
    print("\nPayroll Events Summary:")
    print(payroll_df[['timestamp', 'description', 'success']].describe())
    
    # Enhanced visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Success rate pie chart
    payroll_df['success'].value_counts().plot(
        kind='pie', autopct='%1.1f%%', 
        title='Payroll Success Rate', ax=ax1)
    ax1.set_ylabel('')
    
    # Payroll events over time
    if len(payroll_df) > 1:
        payroll_df.set_index('timestamp', inplace=True)
        payroll_df['success'].resample('D').mean().plot(
            kind='line', marker='o', ax=ax2,
            title='Daily Payroll Success Rate Over Time')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charts/payroll_analysis_enhanced.png')
    plt.close()
    
    return sim_df, payroll_df

def correlation_analysis():
    """Enhanced correlation analysis between different metrics"""
    print("\n=== Analyzing Correlations ===")
    # Get morale data
    morale_df = get_table_as_df("MoraleEvents")
    
    if morale_df.empty:
        print("No morale data available for correlation analysis")
        return pd.DataFrame()
    
    morale_df['timestamp'] = pd.to_datetime(morale_df['timestamp'])
    morale_df.set_index('timestamp', inplace=True)
    morale_df = morale_df[['current_morale']].resample('1min').mean().ffill()
    
    # Get production data
    prod_df = get_table_as_df("ProductionEvents")
    
    if prod_df.empty:
        print("No production data available for correlation analysis")
        return pd.DataFrame()
    
    prod_df['timestamp'] = pd.to_datetime(prod_df['timestamp'])
    prod_df = prod_df[prod_df['event_type'] == 'Harvest Success']
    prod_df.set_index('timestamp', inplace=True)
    hourly_prod = prod_df['quantity'].resample('1min').sum()
    
    # Combine data
    combined_df = pd.concat([morale_df, hourly_prod], axis=1)
    combined_df.columns = ['morale', 'production']
    combined_df['production'] = combined_df['production'].fillna(0)
    
    # Enhanced correlation analysis
    correlation, p_value = stats.pearsonr(combined_df['morale'], combined_df['production'])
    print(f"\nCorrelation between morale and production: {correlation:.2f}")
    print(f"P-value: {p_value:.4f}")
    
    # Enhanced visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Time series plot
    ax1.plot(combined_df.index, combined_df['morale'], label='Morale', color='tab:blue')
    ax1.set_ylabel('Morale', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xlabel('Time')
    
    ax2_twin = ax1.twinx()
    ax2_twin.plot(combined_df.index, combined_df['production'], label='Production', color='tab:red')
    ax2_twin.set_ylabel('Production', color='tab:red')
    ax2_twin.tick_params(axis='y', labelcolor='tab:red')
    
    ax1.set_title(f'Morale and Production Over Time\n(Correlation: {correlation:.2f}, p-value: {p_value:.4f})')
    
    # Scatter plot with regression line
    if len(combined_df) > 2:  # Only plot if we have enough data points
        sns.regplot(x='morale', y='production', data=combined_df, ax=ax2,
                   scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        ax2.set_title('Morale vs. Production Scatter Plot')
        ax2.set_xlabel('Morale Level')
        ax2.set_ylabel('Production Quantity')
    else:
        ax2.text(0.5, 0.5, 'Not enough data points for scatter plot', 
                ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('charts/morale_production_correlation_enhanced.png')
    plt.close()
    
    return combined_df

def generate_full_report():
    """Generate a comprehensive report with all enhanced analyses"""
    print("\n=== Cartel Simulation Analytics Report ===")
    print(f"Report generated: {datetime.now()}\n")
    
    # Check if database exists
    try:
        with open("cartel_simulation.db", 'r'):
            pass
    except FileNotFoundError:
        print("Error: Database file 'cartel_simulation.db' not found")
        return
    
    # Run enhanced analyses
    print("Running enhanced analyses...")
    prod_df = analyze_production()
    process_df = analyze_processing()
    morale_df = analyze_morale()
    sabotage_df = analyze_sabotage()
    sim_df, payroll_df = analyze_financials()
    combined_df = correlation_analysis()
    
    # Enhanced report summary
    print("\n=== Enhanced Report Summary ===")
    if not prod_df.empty:
        success_rate = len(prod_df[prod_df['event_type'] == 'Harvest Success'])/len(prod_df)
        print(f"- Production: {len(prod_df)} events, {success_rate:.1%} success rate")
        
    if not process_df.empty:
        process_success = process_df['success'].mean()
        print(f"- Processing: {len(process_df)} events, {process_success:.1%} success rate")
        
    if not morale_df.empty:
        final_morale = morale_df['current_morale'].iloc[-1]
        morale_change = final_morale - morale_df['current_morale'].iloc[0]
        print(f"- Morale: Started at {morale_df['current_morale'].iloc[0]:.2f}, "
              f"ended at {final_morale:.2f} (Δ{morale_change:.2f})")
        
    if sabotage_df is not None and not sabotage_df.empty:
        sabotage_success = sabotage_df['success'].mean()
        avg_damage = sabotage_df['damage_estimate'].mean()
        print(f"- Sabotage: {len(sabotage_df)} events, {sabotage_success:.1%} success rate, "
              f"average damage {avg_damage:.2f}")
              
    if not sim_df.empty:
        status = sim_df['status'].iloc[0]
        funds = sim_df['final_funds'].iloc[0]
        duration = pd.to_datetime(sim_df['end_time'].iloc[0]) - pd.to_datetime(sim_df['start_time'].iloc[0])
        print(f"- Simulation: Status '{status}', final funds ${funds:,.2f}, duration {duration}")
    
    print("\nEnhanced analysis complete. Visualizations saved to 'charts' directory.")

if __name__ == "__main__":
    generate_full_report()