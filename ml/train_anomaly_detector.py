import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'speed_kmh',
    'fuel_level_pct',
    'fuel_drop_since_last',
    'distance_from_route_km',
    'speed_change_rate',
    'time_since_last_reading_mins'
]

def load_and_preprocess_data(data_path):
    """Loads raw telemetry data and computes features for anomaly detection."""
    logger.info(f"Loading telemetry data from {data_path}")
    df = pd.read_csv(data_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        logger.warning("No timestamp column found, creating a dummy timeline.")
        df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='5T')
        
    # Standardize basic column names
    if 'speed' in df.columns and 'speed_kmh' not in df.columns:
        df = df.rename(columns={'speed': 'speed_kmh'})
    if 'fuel_level' in df.columns and 'fuel_level_pct' not in df.columns:
        df = df.rename(columns={'fuel_level': 'fuel_level_pct'})
        
    # Must group by shipment to avoid cross-shipment diffs
    if 'shipment_id' in df.columns:
        df = df.sort_values(by=['shipment_id', 'timestamp'])
        grouper = df.groupby('shipment_id')
    else:
        logger.warning("No shipment_id found, treating all data as one shipment.")
        df = df.sort_values(by=['timestamp'])
        grouper = df
        
    # Calculate derived features
    df['fuel_drop_since_last'] = grouper['fuel_level_pct'].diff().fillna(0) * -1 # positive drop
    df['speed_change_rate'] = grouper['speed_kmh'].diff().abs().fillna(0)
    
    if 'shipment_id' in df.columns:
        df['time_diff'] = grouper['timestamp'].diff()
    else:
        df['time_diff'] = df['timestamp'].diff()
        
    df['time_since_last_reading_mins'] = df['time_diff'].dt.total_seconds() / 60.0
    df['time_since_last_reading_mins'] = df['time_since_last_reading_mins'].fillna(0)
    
    # 1. distance_from_route_km: simulate with random noise if missing
    if 'distance_from_route_km' not in df.columns:
        np.random.seed(42)
        # Assuming most readings are close to route (exponential decay)
        df['distance_from_route_km'] = np.random.exponential(scale=0.5, size=len(df))
        
    # Ensure all columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            logger.warning(f"Missing feature {col}, filling with 0.")
            df[col] = 0
            
    return df

def train_and_evaluate_models(df, model_dir):
    """Trains IsolationForest and LOF, and saves the best detector."""
    X = df[FEATURE_COLS]
    
    # 2. Normalize all features using StandardScaler
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, 'anomaly_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved anomaly scaler to {scaler_path}")
    
    # 3a. Train IsolationForest
    logger.info("Training IsolationForest...")
    iso_forest = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
    iso_preds = iso_forest.fit_predict(X_scaled)
    df['is_anomaly_iso'] = (iso_preds == -1).astype(int)
    
    # 3b. Train LocalOutlierFactor (offline eval only)
    logger.info("Training LocalOutlierFactor (offline eval only)...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_preds = lof.fit_predict(X_scaled)
    df['is_anomaly_lof'] = (lof_preds == -1).astype(int)
    
    # 4. Evaluate IsolationForest
    total_anomalies = df['is_anomaly_iso'].sum()
    logger.info(f"Total anomalies detected (IsolationForest): {total_anomalies} out of {len(df)}")
    
    if 'vehicle_type' in df.columns:
        dist = df[df['is_anomaly_iso'] == 1]['vehicle_type'].value_counts()
        logger.info(f"Anomaly distribution by vehicle_type:\n{dist.to_string()}")
        
    if 'shipment_id' in df.columns:
        top_shipments = df[df['is_anomaly_iso'] == 1]['shipment_id'].value_counts().head(5)
        logger.info(f"Top 5 shipments by anomaly count:\n{top_shipments.to_string()}")
        
    # 5. Save IsolationForest
    model_path = os.path.join(model_dir, 'anomaly_detector.pkl')
    joblib.dump(iso_forest, model_path)
    logger.info(f"Saved IsolationForest model to {model_path}")
    
    # 7. Save metadata
    metadata = {
        "model_type": "IsolationForest",
        "feature_cols": FEATURE_COLS,
        "contamination": 0.05,
        "n_estimators": 200,
        "trained_at": datetime.utcnow().isoformat() + "Z"
    }
    metadata_path = os.path.join(model_dir, 'anomaly_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved anomaly metadata to {metadata_path}")

def score_telemetry_point(telemetry_dict, model_dir=None):
    """
    Scores a single telemetry reading for anomalies.
    
    Args:
        telemetry_dict (dict): Dictionary mapping feature names to values.
        model_dir (str, optional): Directory containing the models.
        
    Returns:
        dict: contains is_anomaly, anomaly_score, and anomaly_reason
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
    model_path = os.path.join(model_dir, 'anomaly_detector.pkl')
    scaler_path = os.path.join(model_dir, 'anomaly_scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    df = pd.DataFrame([telemetry_dict])
    
    # Map basic fields if necessary
    if 'speed' in df.columns and 'speed_kmh' not in df.columns:
        df['speed_kmh'] = df['speed']
    if 'fuel_level' in df.columns and 'fuel_level_pct' not in df.columns:
        df['fuel_level_pct'] = df['fuel_level']
        
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
            
    X = df[FEATURE_COLS]
    
    # 2. Apply StandardScaler
    # The output represents the z-score (number of standard deviations from the mean)
    X_scaled = scaler.transform(X)
    
    # Predict
    pred = model.predict(X_scaled)[0]
    is_anomaly = bool(pred == -1)
    
    # 6. anomaly_score
    if hasattr(model, 'decision_function'):
        score = float(model.decision_function(X_scaled)[0])
    else:
        score = float(model.score_samples(X_scaled)[0])
        
    # 6. anomaly_reason
    reason = "Normal"
    if is_anomaly:
        z_scores = X_scaled[0]
        max_abs_z_idx = np.argmax(np.abs(z_scores))
        max_z_val = z_scores[max_abs_z_idx]
        feature_name = FEATURE_COLS[max_abs_z_idx]
        
        # Compare to mean ± 3 std -> absolute z-score > 3
        if max_z_val > 3:
            reason = f"{feature_name} unusually high (z={max_z_val:.2f})"
        elif max_z_val < -3:
            reason = f"{feature_name} unusually low (z={max_z_val:.2f})"
        else:
            reason = f"Unusual combination of features, most notably: {feature_name} (z={max_z_val:.2f})"
            
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": score,
        "anomaly_reason": reason
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'telemetry.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(data_path):
        logger.error(f"Telemetry data not found at {data_path}. Please place telemetry.csv in the data directory.")
        return
        
    try:
        df = load_and_preprocess_data(data_path)
    except Exception as e:
        logger.error(f"Error loading telemetry data: {e}")
        return
        
    logger.info(f"Starting Anomaly Detector training with {len(df)} telemetry points...")
    train_and_evaluate_models(df, model_dir)

if __name__ == '__main__':
    main()
