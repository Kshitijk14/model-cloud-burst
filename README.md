# 🌧️ Cloud Burst Prediction & Forecasting System

## Project Overview

The Cloud Burst Prediction & Forecasting System is an advanced, AI-powered solution that combines historical weather data, real-time IoT sensor inputs, and deep learning models to provide precise, localized predictions of extreme rainfall events. By integrating cutting-edge machine learning techniques with comprehensive meteorological data sources, the system offers a sophisticated early warning mechanism for potential cloud burst incidents.

## 🚀 Key Features

- **Multi-Source Data Integration**
  - Historical data from [Open-Mateo API](https://open-meteo.com/en/docs/historical-weather-api)
  - Real-time sensor data from ESP32 and DHT-11
  - Time-series feature engineering

- **Advanced Machine Learning**
  - CNN-1D neural network for time-series forecasting
  - Supports LSTM and GRU architectures
  - Robust preprocessing and data scaling

- **Real-Time Data Pipeline**
  - Firebase real-time database integration
  - Automated data collection and storage
  - Continuous model inference using DVC

- **Web API**
  - Flask-based prediction server
  - RESTful endpoints for data fetching and prediction
  - Cross-origin resource sharing (CORS) support

## 🛠 Technology Stack

- **Languages**: Python
- **Machine Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Automated Workflow**: Docker, DVC
- **Web Framework**: Flask
- **Database**: Firebase Realtime Database
- **IoT**: ESP32, DHT-11 Sensor

## 🔧 System Architecture

### Data Collection
1. **Hardware Sensors**: 
   - ESP32 microcontroller
   - DHT-11 temperature and humidity sensor
   - Rain sensor
   
2. **Data Sources**:
   - Local IoT sensors
   - Open-Mateo historical weather API

### Preprocessing
1. Sensor data conversion
2. **Time-based Feature Engineering**
  - Convert DateTime index to seconds
  - Generate Cyclical Time Features
    * Day Sin/Cos: Captures daily periodicity
    * Year Sin/Cos: Captures yearly seasonality
  
  *Why Cyclical Time Features?*
  - Preserves circular nature of time
  - Helps model capture seasonal patterns
  - Prevents linear misinterpretation of time
  - Improves model's ability to recognize time-based correlations

3. Sliding window sequence creation
4. Standardization and scaling

### Prediction Model
- LSTM
- GRU
- 1D Convolutional Neural Network

## 📦 Installation

### Prerequisites
- Python 3.8+
- TensorFlow
- Flask
- DVC
- Docker
- Firebase account

### Setup Steps
1. Clone the repository
```bash
git clone https://github.com/Kshitijk14/cloud-burst.git
cd model-cloud-burst
```

2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# For Windows
.\env\Scripts\activate

# For macOS/Linux
source env/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set Firebase API Key in `.env`
```bash
export FIREBASE_API_KEY=your_firebase_api_key
```

5. Run the Flask Application
```bash
python app.py
```

## 🌐 Repository Structure

- **Front-End Repository**: [Cloud Burst UI](https://github.com/Kshitijk14/cloud-burst)
  - Contains user interface components
  - Visualization dashboards
  - Client-side application

## 🌐 API Endpoints

- `GET /`: Health check
- `GET /health`: Server status
- `POST /predict`: Make rainfall predictions
- `GET /fetch`: Fetch and process real-time data

## 📊 Model Performance Metrics

- Prediction Window: Hourly
- Prediction Horizon: Next 8 hours
- Metrics: MAE, RMSE, R²

## 🔬 Future Enhancements
- Remote sensing integration for satellite imagery data
- Multi-location support
- Enhanced feature engineering
- Ensemble model approach
- Improved sensor calibration
- Advanced climate pattern analysis

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests.

## 📄 License

This project is licensed under the MIT License.
