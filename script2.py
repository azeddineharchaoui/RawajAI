import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_weather_dataset():
    # Set the date range for the previous year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start_date, end_date)
    
    # Initialize lists for each column
    dates = []
    temperatures = []
    precipitations = []
    wind_speeds = []
    
    # Base parameters for seasonal variation
    base_temp = {
        'winter': 5.0,   # Celsius
        'spring': 15.0,
        'summer': 25.0,
        'fall': 14.0
    }
    
    base_precip = {
        'winter': 0.3,   # Probability of precipitation
        'spring': 0.4,
        'summer': 0.2,
        'fall': 0.35
    }
    
    base_wind = {
        'winter': 15.0,  # km/h
        'spring': 12.0,
        'summer': 8.0,
        'fall': 10.0
    }
    
    for single_date in date_range:
        # Determine season
        if single_date.month in [12, 1, 2]:
            season = 'winter'
        elif single_date.month in [3, 4, 5]:
            season = 'spring'
        elif single_date.month in [6, 7, 8]:
            season = 'summer'
        else:
            season = 'fall'
        
        # Generate temperature with seasonal pattern and daily variation
        temp_variation = np.random.normal(0, 5)  # Daily variation
        temperature = base_temp[season] + temp_variation
        
        # Generate precipitation (correlated with temperature)
        precip_prob = base_precip[season] * (1 + 0.5 * np.sin(single_date.day / 5))  # Some monthly pattern
        precipitation = 0
        
        if random.random() < precip_prob:
            # More precipitation when temperature is moderate
            if 5 < temperature < 20:
                precipitation = np.random.gamma(shape=2, scale=3)  # mm
            else:
                precipitation = np.random.gamma(shape=1.5, scale=2)  # mm
        
        # Generate wind speed (slightly higher when precipitation occurs)
        wind_speed = max(0, np.random.normal(base_wind[season], 5))
        if precipitation > 0:
            wind_speed *= 1.3  # Increase wind speed when it's raining
        
        # Add some extreme weather events (5% chance)
        if random.random() < 0.05:
            if season == 'winter':
                temperature -= 10
                wind_speed += 15
                precipitation += 5
            elif season == 'summer':
                temperature += 8
                wind_speed += 10
                if random.random() < 0.7:
                    precipitation += 15  # Summer storm
        
        # Cap extreme values
        temperature = max(-20, min(40, temperature))
        precipitation = min(100, precipitation)  # Max 100mm in a day
        wind_speed = min(120, wind_speed)  # Max 120 km/h
        
        # Append to lists
        dates.append(single_date.strftime('%Y-%m-%d'))
        temperatures.append(round(temperature, 1))
        precipitations.append(round(precipitation, 1))
        wind_speeds.append(round(wind_speed, 1))
    
    # Create DataFrame
    weather_data = pd.DataFrame({
        'date': dates,
        'temperature': temperatures,
        'precipitation': precipitations,
        'wind_speed': wind_speeds
    })
    
    return weather_data

# Generate the dataset
weather_df = generate_weather_dataset()

# Save to CSV
weather_df.to_csv('weather_data.csv', index=False)

print("Weather dataset generated with 365 days of data:")
print(weather_df.head())
print("\nSummary statistics:")
print(weather_df.describe())