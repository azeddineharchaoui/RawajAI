import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_large_supplychain_data(num_rows=3000):
    # Product categories with IDs from script.py
    product_categories = {
        'LAP': ['Laptop', ['LAP-100', 'LAP-200', 'LAP-300']],
        'PHN': ['Smartphone', ['PHN-100', 'PHN-200']],
        'TAB': ['Tablet', ['TAB-100', 'TAB-200']],
        'MON': ['Monitor', ['MON-100']],
        'PRT': ['Printer', ['PRT-100', 'PRT-200']],
        'HDD': ['Hard Drive', ['HDD-100', 'HDD-200']],
        'SSD': ['SSD', ['SSD-100']],
        'CPU': ['Processor', ['CPU-100']],
        'GPU': ['Graphics Card', ['GPU-100', 'GPU-200']],
        'RAM': ['Memory Module', ['RAM-100']],
        'PSU': ['Power Supply', ['PSU-100']],
        'MB': ['Motherboard', ['MB-100', 'MB-200']],
        'KB': ['Keyboard', ['KB-100']],
        'MS': ['Mouse', ['MS-100', 'MS-200']],
        'NC': ['Network Card', ['NC-100']]
    }
    
    locations = ['Casablanca', 'Rabat', 'Tanger', 'Meknes', 'Fes', 'Agadir', 'Oujda', 'Marrakesh']
    
    # Generate dates for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    data = []
    for _ in range(num_rows):
        random_date = start_date + (end_date - start_date) * random.random()
        product_code, product_info = random.choice(list(product_categories.items()))
        product_name = product_info[0]
        product_id = random.choice(product_info[1])
        location = random.choice(locations)
        
        base_demand = {'LAP': 50, 'PHN': 80, 'TAB': 30, 'MON': 25, 'PRT': 15,
                      'HDD': 40, 'SSD': 45, 'CPU': 20, 'GPU': 15, 'RAM': 35,
                      'PSU': 10, 'MB': 12, 'KB': 25, 'MS': 30, 'NC': 8}[product_code]
        demand = max(0, int(np.random.normal(base_demand, base_demand/3)))
        
        inventory = max(0, demand + random.randint(-20, 50))
        base_cost = {'LAP': 800, 'PHN': 600, 'TAB': 300, 'MON': 200, 'PRT': 150,
                    'HDD': 80, 'SSD': 120, 'CPU': 250, 'GPU': 500, 'RAM': 100,
                    'PSU': 70, 'MB': 150, 'KB': 30, 'MS': 20, 'NC': 40}[product_code]
        cost = round(base_cost * random.uniform(0.95, 1.05), 2)
        lead_time = random.choice([3, 5, 7, 10, 14])
        
        data.append([
            random_date.strftime('%Y-%m-%d'),
            product_id,
            product_name,
            location,
            demand,
            inventory,
            cost,
            lead_time
        ])
    
    df = pd.DataFrame(data, columns=[
        'date', 'product_id', 'product_name', 'location', 
        'demand', 'inventory', 'cost', 'lead_time'
    ])
    return df

# Category mapping from script.py product names to trends_data.csv column names
category_mapping = {
    'Laptop': 'laptop',
    'Smartphone': 'smartphone',
    'Tablet': 'tablet',
    'Monitor': 'monitor',
    'Printer': 'printer',
    'Hard Drive': 'hard drive',
    'SSD': 'ssd',
    'Processor': 'processor',
    'Graphics Card': 'graphics card',
    'Memory Module': 'memory',
    'Power Supply': 'power supply',
    'Motherboard': 'motherboard',
    'Keyboard': 'keyboard',
    'Mouse': 'mouse',
    'Network Card': 'network card'
}

# Generate the supply chain dataset
large_dataset = generate_large_supplychain_data(3000)

# Save the supply chain dataset to CSV
large_dataset.to_csv('supply_chain_dataset.csv', index=False)

# Convert date to datetime
large_dataset['date'] = pd.to_datetime(large_dataset['date'])

# Map product names to trend categories
large_dataset['trend_category'] = large_dataset['product_name'].map(category_mapping)

# Filter for relevant categories (only those we want in trends data)
trend_categories = ['laptop', 'smartphone', 'tablet', 'monitor', 'printer', 
                   'hard drive', 'ssd', 'processor', 'graphics card', 'memory']
filtered_dataset = large_dataset[large_dataset['trend_category'].isin(trend_categories)]

# Aggregate demand by date and trend category
agg_demand = filtered_dataset.groupby(['date', 'trend_category'])['demand'].sum().reset_index()

# Define the date range for the previous year (July 25, 2024 to July 24, 2025)
start_date = pd.to_datetime('2024-07-25')
end_date = pd.to_datetime('2025-07-24')
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Create a complete date-category grid
all_categories = pd.DataFrame({'trend_category': trend_categories})
date_category_grid = pd.MultiIndex.from_product(
    [all_dates, trend_categories],
    names=['date', 'trend_category']
).to_frame(index=False)

# Merge with actual data to fill in values
merged = date_category_grid.merge(agg_demand, on=['date', 'trend_category'], how='left')
merged['demand'] = merged['demand'].fillna(0)

# Pivot the data to have dates as index and trend categories as columns
pivoted = merged.pivot(index='date', columns='trend_category', values='demand')

# Normalize each category to a 0-100 scale based on its maximum demand
max_values = pivoted.max()
trends = (pivoted / max_values * 100).round().astype(int)

# Reset index and format date
trends = trends.reset_index()
trends['date'] = trends['date'].dt.strftime('%Y-%m-%d')

# Define the desired column order to match trends_data.csv
desired_columns = ['date', 'smartphone', 'laptop', 'tablet', 'monitor', 'printer', 
                  'hard drive', 'ssd', 'processor', 'graphics card', 'memory']
trends = trends[desired_columns]

# Save to CSV
trends.to_csv('trends_data.csv', index=False)

print("Successfully generated supply_chain_dataset.csv and trends_data.csv")
print("\nFirst 5 rows of trends_data.csv:")
print(trends.head())