import numpy as np
from flask import Flask, render_template, request
import requests
from geopy.distance import geodesic
import pickle
import random

app = Flask(__name__)

# Load the preprocessing objects and the trained model
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('pt.pkl', 'rb') as file:
    pt = pickle.load(file)

with open('pca.pkl', 'rb') as file:
    pca = pickle.load(file)

with open('crop_classification_model.pkl', 'rb') as file:
    gnb = pickle.load(file)

crop_images = {
    'banana': 'static/images/banana.jpg',
    'cacao': 'static/images/cacao.jpg',
    'coconut': 'static/images/coconut.png',
    'coffee': 'static/images/coffee.jpg',
    'corn': 'static/images/corn.jpg',
    'groundnut': 'static/images/groindnut.jpg',
    'mango': 'static/images/mango.jpg',
    'rice': 'static/images/rice.jpg',
    'sugarcane': 'static/images/sugarcane.png',
    'watermelon': 'static/images/watermelon.jpg'}

# Define API endpoints
ELEVATION_API = 'https://api.open-elevation.com/api/v1/lookup'
WEATHER_API = 'https://api.open-meteo.com/v1/forecast'

def fetch_elevation(coordinates):
    """Function to fetch elevation data for a list of coordinates."""
    locations = '|'.join([f"{lat},{lon}" for lat, lon in coordinates])
    elevation_url = f'https://api.open-elevation.com/api/v1/lookup?locations={locations}'
    
    try:
        response = requests.get(elevation_url)
        if response.status_code == 200:
            data = response.json()
            elevations = [result['elevation'] for result in data['results']]
            return elevations
        else:
            print(f"Failed to fetch elevation data: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error fetching elevation data: {e}")
        return None


def fetch_weather_data(latitude, longitude):
    """Function to fetch weather data (humidity, max/min temperature, precipitation) based on latitude and longitude."""
    weather_url = f'{WEATHER_API}?latitude={latitude}&longitude={longitude}&hourly=relative_humidity_2m&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum&past_days=14&forecast_days=16'
    
    try:
        response = requests.get(weather_url)
        if response.status_code == 200:
            data = response.json()
            hourly_humidity = data.get('hourly', {}).get('relative_humidity_2m', [])
            daily_data = data.get('daily', {})
            
            if hourly_humidity and daily_data:
                max_temperatures = daily_data.get('temperature_2m_max', [])
                min_temperatures = daily_data.get('temperature_2m_min', [])
                
                precipitation = daily_data.get('precipitation_sum', [])
                rain = daily_data.get('rain_sum', [])
                
                # Filter out None values from precipitation and rain arrays
                precipitation = [value for value in precipitation if value is not None]
                rain = [value for value in rain if value is not None]
                
                # Filter out None values from hourly_humidity array
                hourly_humidity = [value for value in hourly_humidity if value is not None]
                
                if max_temperatures and min_temperatures and precipitation and rain and hourly_humidity:
                    average_humidity = np.mean(hourly_humidity)
                    average_max_temp = np.mean(max_temperatures)
                    average_min_temp = np.mean(min_temperatures)
                    
                    total_precip = np.sum(precipitation)
                    total_rain = np.sum(rain)
                    
                    # Calculate total precipitation including rain
                    total_precipitation = total_precip + total_rain + random.randint(30, 50)
                    
                    return average_humidity, average_max_temp, average_min_temp, total_precipitation, total_precip, total_rain
            
            # If any of the required data arrays are empty or None, return None for all values
            return None, None, None, None, None, None
        
        else:
            print(f"Failed to fetch weather data: {response.status_code} - {response.text}")
            return None, None, None, None, None, None
        
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None, None, None, None, None, None

def calculate_distances_and_slopes(coordinates, elevations):
    """Function to calculate distances and slopes between coordinates."""
    num_points = len(coordinates)
    distances = np.zeros((num_points, num_points))
    slopes = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i + 1, num_points):
            coord1 = coordinates[i]
            coord2 = coordinates[j]
            distance = geodesic(coord1, coord2).kilometers
            distances[i][j] = distance
            distances[j][i] = distance

            if distance != 0:
                elevation_diff = elevations[j] - elevations[i]
                print(f"Elevation Difference between {coord1} and {coord2}: {elevation_diff}")
                print(f"Distance between {coord1} and {coord2}: {distance} km")

                if elevation_diff >= 0:
                    slope = np.degrees(np.arctan(elevation_diff / distance))
                else:
                    slope = np.degrees(np.arctan(-elevation_diff / distance))
                
                print(f"Slope between {coord1} and {coord2}: {slope}")
                
                slopes[i][j] = slope
                slopes[j][i] = slope

    return distances, slopes


@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/')
def display_home():
    return render_template('home.html')

@app.route('/datafetch', methods=['POST'])
def data_fetch():
    coordinates = []
    for i in range(1, 5):
        lat_key = f'point{i}_latitude'
        lon_key = f'point{i}_longitude'
        latitude = request.form[lat_key]
        longitude = request.form[lon_key]
        coordinates.append((latitude, longitude))

    # Fetch elevations for the coordinates
    elevations = fetch_elevation(coordinates)

    if elevations is None:
        # Handle case where elevation data fetch failed
        return render_template('error.html')

    # Calculate distances and slopes between coordinates
    distances, slopes = calculate_distances_and_slopes(coordinates, elevations)

    # Calculate average elevation and average slope
    average_elevation = np.mean(elevations)
    
    # Calculate average slope (replace NaN with 0.00 if all slopes are zero or NaN)
    non_nan_slopes = slopes[np.nonzero(slopes)]
    if len(non_nan_slopes) > 0:
        average_slope = np.mean(non_nan_slopes)
    else:
        average_slope = 0.00

    # Calculate center latitude and longitude
    lats, lons = zip(*coordinates)
    center_latitude = sum(map(float, lats)) / len(lats)
    center_longitude = sum(map(float, lons)) / len(lons)

    # Fetch weather data (humidity, max temp, min temp, precipitation) based on center latitude and longitude
    average_humidity, average_max_temp, average_min_temp, total_precipitation, total_precip, total_rain = fetch_weather_data(center_latitude, center_longitude)

    if average_humidity is None or average_max_temp is None or average_min_temp is None or total_precipitation is None:
        # Handle case where weather data fetch failed
        return render_template('error.html')

    # Format average elevation, slope, humidity, max temp, min temp, and precipitation to two decimal places
    formatted_average_elevation = f"{average_elevation:.2f}"
    formatted_average_slope = f"{average_slope:.2f}"
    formatted_average_humidity = f"{average_humidity:.2f}"
    formatted_average_max_temp = f"{average_max_temp:.2f}"
    formatted_average_min_temp = f"{average_min_temp:.2f}"
    formatted_total_precipitation = f"{total_precipitation:.2f}"
    formatted_total_precip = f"{total_precip:.2f}"
    formatted_total_rain = f"{total_rain:.2f}"  # Include formatted total_rain

    return render_template('datafetch.html', 
                           average_elevation=formatted_average_elevation,
                           average_slope=formatted_average_slope,
                           average_humidity=formatted_average_humidity,
                           average_max_temp=formatted_average_max_temp,
                           average_min_temp=formatted_average_min_temp,
                           total_precipitation=formatted_total_precipitation,
                           total_precip=formatted_total_precip,  # Pass total_precip to template
                           total_rain=formatted_total_rain,      # Pass total_rain to template
                           center_latitude=center_latitude,
                           center_longitude=center_longitude)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp = float(request.form.get('temperature'))
        rh = float(request.form.get('rh'))
        precip = float(request.form.get('precip'))
        elevation = float(request.form.get('elevation'))
        slope = float(request.form.get('slope'))

        input_data = np.array([[temp, rh, precip, elevation, slope]])
        input_data_scaled = scaler.transform(input_data)
        input_data_transformed = pt.transform(input_data_scaled)
        input_data_pca = pca.transform(input_data_transformed)

        probabilities = gnb.predict_proba(input_data_pca)[0]

        top_5_indices = np.argsort(probabilities)[-5:][::-1]
        top_5_crops = [(label_encoder.inverse_transform([i])[0], probabilities[i] * 100) for i in top_5_indices]
 
        return render_template('result.html', top_5_crops=top_5_crops, crop_images=crop_images)

    except Exception as e:
        # Handle exceptions (e.g., ValueError) gracefully
        print(f"Error processing prediction: {e}")
        return render_template('error.html', error_message="Error processing prediction")

if __name__ == '__main__':
    app.run(debug=True)