import os
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class FirebaseClient:
    def __init__(self, api_key):
        self.base_url = "https://cloudburst-993f5-default-rtdb.firebaseio.com/.json"
        self.api_key = api_key

    def fetch_data(self):
        """Fetch real-time data from Firebase"""
        try:
            params = {"auth": self.api_key}
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Firebase data fetch error: {str(e)}")
            raise


class DataManager:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.ensure_directory_exists()

    def ensure_directory_exists(self):
        """Ensure the directory for the CSV file exists"""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def process_firebase_data(self, firebase_data):
        """Process raw Firebase data into a DataFrame"""
        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            new_data = {
                'DateTime': [current_datetime],
                'Humidity': [firebase_data['DHT_11']['Humidity']],
                'Temperature': [firebase_data['DHT_11']['Temperature']],
                'Rain': [firebase_data['Precipitation']['Rain']]
            }
            
            return pd.DataFrame(new_data)
        except KeyError as e:
            print(f"Error processing Firebase data: {str(e)}")
            raise ValueError(f"Invalid Firebase data structure: {str(e)}")

    def update_csv(self, new_data_df):
        """Update the CSV file with new data"""
        try:
            if self.csv_path.exists():
                existing_data_df = pd.read_csv(self.csv_path)
                updated_data_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
            else:
                updated_data_df = new_data_df
            
            updated_data_df.to_csv(self.csv_path, index=False)
            return updated_data_df
        except Exception as e:
            print(f"Error updating CSV: {str(e)}")
            raise


class RainPredictionApp:
    def __init__(self):
        self.base_dir = Path('artifacts')
        self.model_path = self.base_dir / 'models/hardware_model/model_cnn_1d.keras'
        self.scaler_path = self.base_dir / 'scalers/hardware_model/scaler.save'
        self.data_path = self.base_dir / 'dataset/fetched_hardware_data.csv'
        self.window_size = 8
        
        # Initialize components
        self.firebase_client = FirebaseClient(os.getenv('FIREBASE_API_KEY'))
        self.data_manager = DataManager(self.data_path)
        
        # Load model and scaler
        self.load_artifacts()
        
        # Initialize global variables
        self.rain_predict_mean = None
        self.rain_predict_std = None

    def load_artifacts(self):
        """Load the model and scaler artifacts"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}")
            
            print("Loading model and scaler...")
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
                raise RuntimeError("Loaded scaler is not properly fitted")
                
            print("Model and scaler loaded successfully")
            
        except Exception as e:
            print(f"Error loading artifacts: {str(e)}")
            raise

    def preprocess_dataframe(self, df):
        """Preprocess the input dataframe"""
        try:
            # Convert Rain sensor values to rain_mm
            df['Rain_mm'] = 0.0075 * (4095 - df['Rain'])
            df.drop(columns=['Rain'], inplace=True)

            # Process datetime
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H:%M', dayfirst=True)
            df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
            df.index = df['DateTime']
            df.drop(columns=['DateTime'], inplace=True)

            # Rename columns
            df = df.rename(columns={
                'Humidity': 'relative_humidity_2m',
                'Temperature': 'temperature_2m',
                'Rain_mm': 'rain'
            })

            # Add time features
            day = 24 * 60 * 60
            year = 365.2425 * day
            seconds = df.index.map(pd.Timestamp.timestamp)
            
            df['Day sin'] = np.sin(seconds * (2 * np.pi / day))
            df['Day cos'] = np.cos(seconds * (2 * np.pi / day))
            df['Year sin'] = np.sin(seconds * (2 * np.pi / year))
            df['Year cos'] = np.cos(seconds * (2 * np.pi / year))
            
            rain = df['rain'].copy()
            df.drop(columns=['rain'], inplace=True)
            df['Rain'] = rain

            return df
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def df_to_sequences(self, df):
        """Convert dataframe to sequences"""
        try:
            df_as_np = df.to_numpy()
            X = []
            for i in range(len(df_as_np) - self.window_size):
                row = [r for r in df_as_np[i:i + self.window_size]]
                X.append(row)
            return np.array(X)
            
        except Exception as e:
            print(f"Error in sequence conversion: {str(e)}")
            raise

    def standardize_and_scale(self, X):
        """Standardize and scale the input data"""
        try:
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Standardize rain data
            rain_data = X[:, :, 0]
            self.rain_predict_mean = np.mean(rain_data[~np.isnan(rain_data)])
            self.rain_predict_std = np.std(rain_data[~np.isnan(rain_data)])
            
            if self.rain_predict_std == 0:
                self.rain_predict_std = 1
                
            X[:, :, 0] = (X[:, :, 0] - self.rain_predict_mean) / self.rain_predict_std
            
            # Scale all features
            num_samples, num_timesteps, num_features = X.shape
            X_reshaped = X.reshape(-1, num_features)
            X_scaled = self.scaler.transform(X_reshaped)
            
            return X_scaled.reshape(num_samples, num_timesteps, num_features)
            
        except Exception as e:
            print(f"Error in standardization and scaling: {str(e)}")
            raise

    def predict(self, data):
        """Make predictions using the loaded model"""
        try:
            df = pd.DataFrame(data)
            demo = df.copy()
            
            # Preprocess
            processed_df = self.preprocess_dataframe(df)
            
            # Convert to sequences
            X = self.df_to_sequences(processed_df)
            
            # Scale
            X = self.standardize_and_scale(X)
            
            # Predict
            predictions = self.model.predict(X)
            
            # Generate timestamps
            last_datetime = pd.to_datetime(demo['DateTime'].iloc[-1])
            prediction_timestamps = [last_datetime + pd.Timedelta(hours=i + 1) for i in range(len(predictions))]
            
            # Format results
            results = [{
                'timestamp': timestamp,
                'rain_prediction_mm': float(pred)
            } for timestamp, pred in zip(prediction_timestamps, predictions.flatten())]
            
            return results
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise

    def fetch_and_process_data(self):
        """Fetch data from Firebase and process it"""
        try:
            # Fetch data from Firebase
            firebase_data = self.firebase_client.fetch_data()
            
            # Process the data
            new_data_df = self.data_manager.process_firebase_data(firebase_data)
            
            # Update CSV
            updated_df = self.data_manager.update_csv(new_data_df)
            
            return {
                "new_data": new_data_df.to_dict(orient='records')[0],
                "total_records": len(updated_df)
            }
        except Exception as e:
            print(f"Error in fetch and process: {str(e)}")
            raise

# Initialize the prediction app
prediction_app = RainPredictionApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """Home endpoint"""
    return jsonify({
        "status": "success",
        "message": "Rain Prediction Server is running"
    })

@app.route("/health", methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "Server is healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    """Prediction endpoint"""
    try:
        # Validate input
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Request must be JSON"
            }), 400
            
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({
                "status": "error",
                "message": "Request must contain 'data' field"
            }), 400
            
        # Make prediction
        predictions = prediction_app.predict(data['data'])
        
        return jsonify({
            "status": "success",
            "predictions": predictions
        })
        
    except Exception as e:
        print(f"Error processing prediction request: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/fetch", methods=['GET'])
@cross_origin()
def fetch_data():
    """Endpoint to fetch and store real-time data from Firebase"""
    try:
        # Validate Firebase API key
        if not os.getenv('FIREBASE_API_KEY'):
            return jsonify({
                "status": "error",
                "message": "Firebase API key not configured"
            }), 500
        
        # Fetch and process data
        result = prediction_app.fetch_and_process_data()
        
        # Generate predictions for the new data
        predictions = prediction_app.predict([result['new_data']])
        
        return jsonify({
            "status": "success",
            "message": "Data fetched and processed successfully",
            "data": {
                "current_reading": result['new_data'],
                "total_records": result['total_records'],
                "predictions": predictions
            }
        })
        
    except requests.RequestException as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to fetch data from Firebase: {str(e)}"
        }), 503
        
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid data structure: {str(e)}"
        }), 422
        
    except Exception as e:
        print(f"Error in fetch endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Ensure required environment variables are set
    if not os.getenv('FIREBASE_API_KEY'):
        print("FIREBASE_API_KEY environment variable not set!")
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)