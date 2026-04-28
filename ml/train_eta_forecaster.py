import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'departure_hour', 'departure_day_of_week', 'departure_month', 'is_weekend',
    'scheduled_duration_hrs', 'weight_class', 'vehicle_type_encoded',
    'avg_speed_kmh', 'min_speed_kmh', 'speed_variance', 'num_stops', 'avg_fuel_level_pct',
    'fuel_drop_rate', 'pct_time_idle', 'route_deviation_km',
    'num_disruptions_on_route', 'max_disruption_severity', 'is_weather_disruption',
    'is_congestion_disruption', 'disruption_overlap_hours', 'current_progress_pct'
]

def load_and_preprocess_data(data_path):
    """Loads features, filters for delivered shipments, and prepares training data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    initial_len = len(df)
    # 1. Drop rows where delay_minutes is null
    df = df.dropna(subset=['delay_minutes'])
    logger.info(f"Dropped {initial_len - len(df)} rows with null delay_minutes.")
    
    # 2. Clip delay_minutes to range [-120, 1440]
    df['delay_minutes'] = df['delay_minutes'].clip(lower=-120, upper=1440)
    logger.info("Clipped delay_minutes to range [-120, 1440].")
    
    # Label encode weight_class if it's string
    if 'weight_class' in df.columns and df['weight_class'].dtype == 'object':
        le = LabelEncoder()
        df['weight_class'] = le.fit_transform(df['weight_class'].astype(str))
        logger.info("Label encoded weight_class.")
        
    # Ensure all required features are present
    for col in FEATURE_COLS:
        if col not in df.columns:
            logger.warning(f"Missing column {col}. Filling with 0.")
            df[col] = 0
            
    # Convert booleans to integers
    for col in FEATURE_COLS:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
            
    X = df[FEATURE_COLS]
    y = df['delay_minutes']
    return X, y

def train_and_evaluate_models(X, y, model_dir):
    """Trains 3 regressor models, evaluates them, and saves the best one by MAE."""
    logger.info("Splitting data into train and test sets (80/20)...")
    # 4. Train/test split 80/20, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Define models
    models = {
        'Ridge': Ridge(random_state=42),
        'XGBRegressor': XGBRegressor(
            n_estimators=300, 
            max_depth=6, 
            learning_rate=0.05, 
            random_state=42
        ),
        'LGBMRegressor': LGBMRegressor(
            n_estimators=300, 
            max_depth=6, 
            learning_rate=0.05,
            random_state=42
        )
    }
    
    best_model_name = None
    best_mae = float('inf')
    best_rmse = float('inf')
    best_model = None
    
    # 6. Evaluate each model
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"{name} Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Track the best model by lowest MAE
        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
            best_model_name = name
            best_model = model
            
    logger.info(f"Best Model: {best_model_name} with MAE: {best_mae:.4f} and RMSE: {best_rmse:.4f}")
    
    # 7. Save the best model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
    model_path = os.path.join(model_dir, 'eta_forecaster.pkl')
    joblib.dump(best_model, model_path)
    logger.info(f"Saved best model to {model_path}")
    
    # 9. Save model metadata
    metadata = {
        "model_type": best_model_name,
        "feature_cols": FEATURE_COLS,
        "mae": float(best_mae),
        "rmse": float(best_rmse),
        "trained_at": datetime.utcnow().isoformat() + "Z"
    }
    
    metadata_path = os.path.join(model_dir, 'eta_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved model metadata to {metadata_path}")

def predict_eta(feature_dict, scheduled_delivery_iso_str, model_dir=None):
    """
    Predicts the delay minutes and ETA datetime for a shipment.
    
    Args:
        feature_dict (dict): Dictionary mapping feature names to values.
        scheduled_delivery_iso_str (str): ISO formatted scheduled delivery datetime.
        model_dir (str, optional): Directory containing the models. Defaults to ml/models.
        
    Returns:
        dict: Prediction results including predicted_delay_minutes, predicted_delivery, and confidence_band_minutes.
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
    model_path = os.path.join(model_dir, 'eta_forecaster.pkl')
    metadata_path = os.path.join(model_dir, 'eta_metadata.json')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
    model = joblib.load(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    df = pd.DataFrame([feature_dict])
    
    # Fill missing features with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
            
    # Fallback for categorical weight_class
    if 'weight_class' in df.columns and df['weight_class'].dtype == 'object':
        weight_map = {'light': 0, 'medium': 1, 'heavy': 2}
        df['weight_class'] = df['weight_class'].map(weight_map).fillna(0)
            
    # Apply scaler if it exists
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            exclude_cols = ['shipment_id', 'is_delayed', 'delay_minutes', 'departure_hour', 
                            'departure_day_of_week', 'departure_month', 'is_weekend', 
                            'weight_class', 'vehicle_type_encoded', 'is_weather_disruption', 
                            'is_congestion_disruption']
            continuous_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
            
            if len(continuous_cols) > 0 and hasattr(scaler, 'n_features_in_'):
                if scaler.n_features_in_ == len(continuous_cols):
                    df[continuous_cols] = scaler.transform(df[continuous_cols])
        except Exception as e:
            logger.warning(f"Could not apply scaler during inference: {e}")
            
    X = df[FEATURE_COLS]
    
    # Predict delay minutes
    pred_delay_minutes = float(model.predict(X)[0])
    
    # Parse scheduled delivery
    try:
        # Handle 'Z' suffix for UTC
        if scheduled_delivery_iso_str.endswith('Z'):
            scheduled_delivery_iso_str = scheduled_delivery_iso_str[:-1] + '+00:00'
        scheduled_delivery = datetime.fromisoformat(scheduled_delivery_iso_str)
    except Exception as e:
        logger.error(f"Error parsing scheduled_delivery_iso_str: {e}")
        raise ValueError(f"Invalid ISO datetime string: {scheduled_delivery_iso_str}")
        
    # Calculate ETA
    predicted_delivery = scheduled_delivery + timedelta(minutes=pred_delay_minutes)
    
    # Calculate confidence band: +/- 1.5 * RMSE
    rmse = metadata.get('rmse', 0)
    confidence_band_minutes = int(1.5 * rmse)
    
    # Convert prediction back to 'Z' suffix format if it's naive or explicit UTC
    out_iso = predicted_delivery.isoformat()
    if "+00:00" in out_iso:
        out_iso = out_iso.replace("+00:00", "Z")
    elif out_iso.endswith('Z') is False and predicted_delivery.tzinfo is None:
        out_iso += "Z" # Assuming UTC context
    
    return {
        "predicted_delay_minutes": int(round(pred_delay_minutes)),
        "predicted_delivery": out_iso,
        "confidence_band_minutes": confidence_band_minutes
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'features.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}. Please run feature_engineering.py to generate it.")
        return
        
    try:
        X, y = load_and_preprocess_data(data_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
        
    logger.info(f"Starting ETA forecaster training and evaluation with {len(X)} samples...")
    train_and_evaluate_models(X, y, model_dir=model_dir)

if __name__ == '__main__':
    main()
