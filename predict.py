import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
from logger import logger, metrics_logger

# define paths using Pathlib
base_dir = Path('artifacts')
data_path = base_dir / 'dataset/fetched_hardware_data.csv'
model_path = base_dir / 'models/hardware_model/model_cnn_1d.keras'
scaler_path = base_dir / 'scalers/hardware_model/scaler.save' 

# Load the model
if model_path.exists():
    logger.info(f"Model file found at {model_path}")
    logger.info("Loading model...")
    model = load_model(model_path)
    logger.info("Model loaded successfully")
    logger.info("************************************")
else:
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the scaler
if scaler_path.exists():
    logger.info(f"Scaler file found at {scaler_path}")
    logger.info("Loading scaler...")
    scaler = joblib.load(scaler_path)
    if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        raise RuntimeError("Loaded scaler is not fitted. Please check the saved scaler file.")
    logger.info("Scaler loaded successfully")
    logger.info("************************************")
else:
    logger.error(f"Scaler file not found at {scaler_path}")
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

# Global variables
WINDOW_SIZE = 8
rain_predict_mean = None
rain_predict_std = None


# Function to preprocess the raw dataframe
def preprocess_dataframe(df):
    """
    Preprocess the raw input dataframe:
    - Convert rain sensor values to millimeters
    - Convert datetime to index
    - Add sine and cosine time features
    - Rename columns
    """
    try:
        logger.debug("Starting dataframe preprocessing")
        
        # Convert Rain sensor values to rain_mm
        df['Rain_mm'] = 0.0075 * (4095 - df['Rain'])
        df.drop(columns=['Rain'], inplace=True)
        logger.debug(f"Rain conversion complete. Range: {df['Rain_mm'].min():.4f} to {df['Rain_mm'].max():.4f}")

        # Convert datetime to to correct format as per Open-Mateo Dataset
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H:%M', dayfirst=True)
        df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')

        # Convert DateTime to index
        # df.index = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
        df.index = df['DateTime']
        df.drop(columns=['DateTime'], inplace=True)

        # Rename columns
        df = df.rename(columns={
            'Humidity': 'relative_humidity_2m',
            'Temperature': 'temperature_2m',
            'Rain_mm': 'rain'
        })

        # Add sine and cosine time features
        rain = df['rain']
        df2 = pd.DataFrame({'Rain': rain})
        day = 24 * 60 * 60
        year = 365.2425 * day
        df2['Seconds'] = df2.index.map(pd.Timestamp.timestamp)
        df2['Day sin'] = np.sin(df2['Seconds'] * (2 * np.pi / day))
        df2['Day cos'] = np.cos(df2['Seconds'] * (2 * np.pi / day))
        df2['Year sin'] = np.sin(df2['Seconds'] * (2 * np.pi / year))
        df2['Year cos'] = np.cos(df2['Seconds'] * (2 * np.pi / year))
        df2 = df2.drop(columns=['Seconds'])
        
        # Combine original dataframe with time features
        df = pd.concat([df, df2], axis=1)
        df.drop(columns=['rain'], inplace=True)
        
        logger.info(f"Preprocessed DataFrame shape: {df.shape}")
        logger.info(f"Preprocessed DataFrame columns: {df.columns.tolist()}")
        logger.info("Preprocessing completed successfully!")
        logger.info("************************************")
        
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise


# Function to transform dataframe into sequences
def df_to_X_y(df, window_size=WINDOW_SIZE):
    """
    Convert the dataframe into sequences of X (features) and y (target variable).
    """
    try:
        logger.info("Starting dataframe to sequences conversion")
        df_as_np = df.to_numpy()
        X, y = [], []
        for i in range(len(df_as_np) - window_size):
            row = [r for r in df_as_np[i: i + window_size]]
            X.append(row)
            label = df_as_np[i + window_size][0]  # Target variable is the first column (rain)
            y.append(label)
        
        logger.info("Converting dataframe into sequences completed successfully!")
        logger.info("************************************")
        
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"Error during dataframe to sequences conversion: {str(e)}")
        raise


# Function to standardize and scale input data
def standardize_and_scale(X, scaler):
    """
    Standardize and scale input data.
    - If `fit_scaler` is True, fit the scaler. Otherwise, transform using pre-fitted scaler.
    """
    try:
        logger.info("Starting standardization and scaling")
        global rain_predict_mean, rain_predict_std
        
        # Handle potential NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Print the range of values before standardization
        logger.info(f"Input data range: min={X.min()}, max={X.max()}")
        
        # Standardize the rain column
        rain_data = X[:, :, 0]
        rain_predict_mean = np.mean(rain_data[~np.isnan(rain_data)])
        rain_predict_std = np.std(rain_data[~np.isnan(rain_data)])
        
        print(f"Rain statistics - mean: {rain_predict_mean:.4f}, std: {rain_predict_std:.4f}")
        
        if rain_predict_std == 0:
            rain_predict_std = 1  # Prevent division by zero
        
        X[:, :, 0] = (X[:, :, 0] - rain_predict_mean) / rain_predict_std
        
        logger.info("Standardization completed successfully!")
        logger.info("************************************")

        # Reshape for scaling
        num_samples, num_timesteps, num_features = X.shape
        X_reshaped = X.reshape(-1, num_features)

        # Transform using the pre-fitted scaler
        X_scaled = scaler.transform(X_reshaped)
        
        # Print the range of scaled values
        logger.info(f"Scaled data range: min={X_scaled.min()}, max={X_scaled.max()}")

        # Reshape back to original dimensions
        X = X_scaled.reshape(num_samples, num_timesteps, num_features)
        
        logger.info("Scaling completed successfully!")
        logger.info("************************************")
        
        return X
    except Exception as e:
        logger.error(f"Error during standardization and scaling: {str(e)}")
        raise


# Function to make predictions
def predict_next_day(df, window_size=WINDOW_SIZE):
    """
    Make hourly predictions for the next day based on input dataframe.
    """
    try:
        logger.info("Starting prediction")
        
        # Log input data metrics
        metrics_logger.info(
            f"{df.shape[0]}x{df.shape[1]},"
            f"{df['Rain'].min():.4f},"
            f"{df['Rain'].max():.4f},"
            f"{df['Rain'].mean():.4f},"
            f"{df['Rain'].std():.4f},"
            f"0,0,0"  # Placeholder for prediction metrics
        )
        
        # Create a copy of the dataframe and assign it to demo
        demo = df.copy()
        
        # Preprocess the dataframe
        processed_df = preprocess_dataframe(df)
        
        # Print the range of values in processed data
        logger.info("\nProcessed data ranges:")
        for col in processed_df.columns:
            logger.info(f"{col}: min={processed_df[col].min():.4f}, max={processed_df[col].max():.4f}")

        # Convert dataframe into sequences
        X, _ = df_to_X_y(processed_df, window_size=window_size)

        # Standardize and scale the input
        X = standardize_and_scale(X, scaler)

        # Make predictions
        predictions = model.predict(X)
        
        # Print raw predictions before inverse standardization
        logger.info("\nRaw predictions (before inverse standardization):")
        logger.info(predictions[:5])
        
        # Generate timestamps for predictions
        last_datetime = pd.to_datetime(demo['DateTime'].iloc[-1])
        prediction_timestamps = [last_datetime + pd.Timedelta(hours=i + 1) for i in range(len(predictions))]
        
        # Create a DataFrame with predictions and timestamps
        prediction_df = pd.DataFrame({
            'DateTime': prediction_timestamps,
            'Rain_Prediction_mm': predictions.flatten()
        })
        
        # Log prediction metrics
        logger.info(f"Generated {len(predictions)} predictions")
        logger.debug(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")

        logger.info("Prediction completed successfully!")
        logger.info("************************************")
        
        return prediction_df
    except Exception as e:
        logger.error(f"Error during prediction ", exc_info=True)
        raise


# Load new data and make predictions
try:
    logger.info("Starting prediction script")
    logger.info("Loading new data...")
    new_df = pd.read_csv(data_path)
    logger.info(f"Data loaded, shape: {new_df.shape}")
    
    logger.info("Making predictions...")
    predictions = predict_next_day(new_df)
    
    logger.info("\nHourly Rain Predictions for the Next Day:")
    logger.info(f"\n{predictions}")
except Exception as e:
    logger.error(f"Critical error in main execution: {str(e)}", exc_info=True)
    raise
finally:
    logger.info("Prediction script completed")
