**GeoCropAI - Crop Recommendation System**

GeoCropAI is an intelligent crop recommendation system designed to provide farmers and agricultural experts with data-driven insights into which crops are best suited for a given location. The system uses geographical data, soil characteristics, and environmental parameters to recommend crops that will thrive in the inputted conditions.

**Table of Contents**

Project Overview

Features

User Interface

Installation

Usage

Dataset

File Structure

Technologies Used

Future Enhancements

Contributors


**Project Overview**

GeoCropAI is designed to assist users by providing crop recommendations based on geographical location and soil data. The system inputs latitude and longitude, analyzes the corresponding soil and weather conditions, and suggests the best crops for the region. This project combines data science and agriculture to deliver actionable insights for improving crop yields.

**Features**

User-friendly interface: A web-based form where users can input latitude and longitude to get crop recommendations.
Geolocation-based crop recommendation: Uses the latitude and longitude to determine suitable crops.
Soil and environmental analysis: The system evaluates soil pH, clay content, sand content, and other parameters to recommend crops.
Detailed result view: Displays recommended crops along with temperature, humidity, rainfall, and other relevant data.
Expandable framework: Designed to integrate more data sources, including real-time weather data in future versions.

**User Interface**

<img width="959" alt="image" src="https://github.com/user-attachments/assets/b5c4afcb-f6df-42ce-a737-3a3633e7df95">

<img width="957" alt="image" src="https://github.com/user-attachments/assets/22bdc835-4c58-48d7-91a6-4e73dca5513a">

<img width="940" alt="image" src="https://github.com/user-attachments/assets/595c7550-6553-4db9-a017-d8c87e8000ee">


**Installation**

Follow these steps to run GeoCropAI on your local machine:

Clone the repository: git clone https://github.com/yegamatimanasa/GeoCropAI.git

cd GeoCropAI

**Install dependencies:**

Flask
Pandas
NumPy
Scikit-learn 
Jinja2

**You can install them using:**

pip install Flask pandas numpy scikit-learn

Run the Flask application: python app.py

Access the application: Open your web browser and navigate to http://127.0.0.1:5000 to access the GeoCropAI system.

**Usage**

Open the web application.
Enter the Latitude and Longitude of the location for which you want a crop recommendation.
Click the "Get Recommendation" button.
View the recommendation result, which includes:
The recommended crop.
Environmental conditions like temperature, humidity, and rainfall.
Soil parameters like pH, clay, sand, and silt content.

**Dataset**

HWSD_DATA.csv: A dataset containing soil data such as pH, clay, sand, and silt content, as well as Cation Exchange Capacity (CEC). This data is used to evaluate the suitability of different crops.

crop_data.csv: Contains a list of crops and related data used for making the recommendation.
These datasets are key components of the recommendation engine, helping to match environmental and soil conditions to the optimal crops.

**File Structure**

<img width="508" alt="image" src="https://github.com/user-attachments/assets/84b249f7-8908-45ee-b0f8-75f6784f7416">

**Technologies Used**

Python: For backend logic and handling data.
Flask: As the web framework to create the application.
Pandas: For data manipulation and analysis.
Jupyter Notebook: Used for data exploration and model development.
HTML/CSS (Bootstrap): For the frontend, including a responsive and clean UI.

**Future Enhancements**

Report Generation: Add the ability to download a detailed report of the crop recommendation.
Weather API Integration: Incorporate real-time weather data to improve recommendation accuracy.
Improved Recommendation Algorithm: Add more factors (e.g., market prices, crop rotations) to enhance the recommendation system.
Mobile Optimization: Further improve the user interface for mobile devices.

**Contributors**
Yegamati Manasa

Feel free to contribute by submitting issues or creating pull requests to enhance the system.








