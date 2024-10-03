from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import json
from geopy.distance import geodesic
from sklearn.impute import SimpleImputer
import numpy as np
import random
import requests

app = Flask(__name__)

# Load the CSV dataset
def load_dataset(csv_file):
    df = pd.read_csv(csv_file, dtype={'MU_GLOBAL': str}, low_memory=False)
    return df

# Load the centroid mapping from a JSON file
def load_centroid_mapping(filepath='mu_global_centroids.json'):
    with open(filepath, 'r') as f:
        return json.load(f)

# Function to get the nearest MU_GLOBAL based on lat/lon
def get_nearest_mu_global(lat, lon, centroids):
    nearest_mu_global = None
    min_distance = float('inf')
    
    for mu_global, (centroid_lat, centroid_lon) in centroids.items():
        distance = geodesic((lat, lon), (centroid_lat, centroid_lon)).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest_mu_global = mu_global
            
    return nearest_mu_global

# Function to train the machine learning model
def train_model(df):
    X = df[["Temperature", "Humidity", "Rainfall_1h", "Rainfall_3h", "pH", "Clay_Content", "Sand_Content", "Silt_Content", "CEC"]]
    y = df["Crop"]
    
    X = X.apply(pd.to_numeric, errors='coerce')
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_imputed, y)
    return model

# Updated function to retrieve soil properties
def get_soil_properties(lat, lon, df, centroids):
    nearest_mu_global = get_nearest_mu_global(lat, lon, centroids)
    
    matching_rows = df[df['MU_GLOBAL'] == nearest_mu_global]
    
    if matching_rows.empty:
        print(f"No matching MU_GLOBAL found for {nearest_mu_global}. Filling with random data.")
        random.seed(nearest_mu_global)
        return {
            "pH": round(random.uniform(4.0, 9.0), 2),
            "Clay_Content": round(random.uniform(5.0, 40.0), 2),
            "Sand_Content": round(random.uniform(10.0, 70.0), 2),
            "Silt_Content": round(random.uniform(5.0, 50.0), 2),
            "CEC": round(random.uniform(5.0, 50.0), 2)
        }

    soil_row = matching_rows.iloc[0]

    def get_value_or_random(val, min_range, max_range):
        if pd.isna(val) or val is None or (isinstance(val, str) and val.strip() == ''):
            return round(random.uniform(min_range, max_range), 2)
        return round(float(val), 2)

    soil_data = {
        "pH": get_value_or_random(soil_row.get("S_PH_H2O"), 4.0, 9.0),
        "Clay_Content": get_value_or_random(soil_row.get("S_CLAY"), 5.0, 40.0),
        "Sand_Content": get_value_or_random(soil_row.get("S_SAND"), 10.0, 70.0),
        "Silt_Content": get_value_or_random(soil_row.get("S_SILT"), 5.0, 50.0),
        "CEC": get_value_or_random(soil_row.get("S_CEC_SOIL"), 5.0, 50.0)
    }
    
    print(f"Soil data for MU_GLOBAL {nearest_mu_global}: {soil_data}")  # Debugging
    return soil_data

# Function to get weather data from Tomorrow.io API
def get_weather_data(lat, lon, api_key):
    url = f"https://api.tomorrow.io/v4/timelines?location={lat},{lon}&fields=temperature,humidity,precipitationIntensity&timesteps=current&units=metric&apikey={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        current_values = data['data']['timelines'][0]['intervals'][0]['values']
        weather_data = {
            "Temperature": current_values.get("temperature", "N/A"),
            "Humidity": current_values.get("humidity", "N/A"),
            "Rainfall_1h": current_values.get("precipitationIntensity", 0),
            "Rainfall_3h": current_values.get("precipitationIntensity", 0) * 3
        }
        return weather_data
    else:
        print(f"Failed to fetch weather data: {response.status_code} - {response.text}")
        return {
            "Temperature": 25.0,
            "Humidity": 60,
            "Rainfall_1h": 0.0,
            "Rainfall_3h": 0.0
        }

# Expanded list of crops
available_crops = [
    "Onion", "Tomato", "Corn", "Wheat", "Potato", "Rice", "Soybean", "Barley", 
    "Lettuce", "Cabbage", "Carrot", "Pepper", "Eggplant", "Spinach", "Peas", 
    "Chickpea", "Garlic", "Broccoli", "Cauliflower", "Radish", "Turnip", 
    "Sweet Potato", "Sugarcane", "Cotton", "Sunflower", "Millet", "Sorghum", 
    "Mustard", "Flax", "Cassava", "Ginger", "Turmeric", "Papaya", "Mango", 
    "Banana", "Grapes", "Apple", "Pear", "Peach", "Pineapple", "Cucumber"
]

# Function to assign unique crops to different centroid keys
def assign_unique_crops(centroids):
    assigned_crops = {}
    available_crops_copy = available_crops.copy()
    
    for centroid_key in centroids.keys():
        if len(available_crops_copy) == 0:
            available_crops_copy = available_crops.copy()  # Reset if crops are exhausted
        random.shuffle(available_crops_copy)
        assigned_crops[centroid_key] = available_crops_copy.pop()
    
    return assigned_crops

# Assign unique crops to centroid keys at app start
assigned_crops = assign_unique_crops(load_centroid_mapping())

# Define the `predict_crop` function to handle random assignment if model is not provided
def predict_crop(model, data, centroid_key):
    if model:
        input_data = pd.DataFrame([data])
        input_data.replace([None], np.nan, inplace=True)
        predicted_crop = model.predict(input_data)
        return predicted_crop[0]
    else:
        # Assign a crop based on the centroid key
        return assigned_crops.get(centroid_key, "Unknown Crop")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    latitude = float(request.form.get('latitude'))
    longitude = float(request.form.get('longitude'))

    df_soil = load_dataset('HWSD_DATA.csv')
    centroids = load_centroid_mapping()

    soil_data = get_soil_properties(latitude, longitude, df_soil, centroids)

    api_key = "vi8STYFBAx1wek3JsRiCyNqnLMgOA6Dy"  # Replace with your actual API key
    weather_data = get_weather_data(latitude, longitude, api_key)

    combined_data = {
        "Temperature": weather_data['Temperature'],
        "Humidity": weather_data['Humidity'],
        "Rainfall_1h": weather_data['Rainfall_1h'],
        "Rainfall_3h": weather_data['Rainfall_3h'],
        "pH": soil_data['pH'],
        "Clay_Content": soil_data['Clay_Content'],
        "Sand_Content": soil_data['Sand_Content'],
        "Silt_Content": soil_data['Silt_Content'],
        "CEC": soil_data['CEC']
    }

    nearest_centroid_key = get_nearest_mu_global(latitude, longitude, centroids)
    recommended_crop = predict_crop(None, combined_data, nearest_centroid_key)

    return render_template('result.html', latitude=latitude, longitude=longitude, 
                           temperature=weather_data['Temperature'], humidity=weather_data['Humidity'], 
                           rainfall_1h=weather_data['Rainfall_1h'], rainfall_3h=weather_data['Rainfall_3h'],
                           crop=recommended_crop, soil_data=soil_data)

if __name__ == '__main__':
    df_soil = load_dataset('HWSD_DATA.csv')
    centroids = load_centroid_mapping()

    df = load_dataset('crop_data.csv')
    model = train_model(df)
    
    app.run(debug=True)
