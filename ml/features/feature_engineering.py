import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def load_data(data_dir='ml/data'):
    """
    Loads the shipments, telemetry, and disruptions data from the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing the CSV files.
        
    Returns:
        tuple: (shipments_df, telemetry_df, disruptions_df)
    """
    shipments = pd.read_csv(os.path.join(data_dir, 'shipments.csv'))
    telemetry = pd.read_csv(os.path.join(data_dir, 'telemetry.csv'))
    disruptions = pd.read_csv(os.path.join(data_dir, 'disruptions.csv'))
    return shipments, telemetry, disruptions

def engineer_shipment_features(shipments):
    """
    Engineers features directly from the shipments dataset.
    
    Args:
        shipments (pd.DataFrame): The raw shipments dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with engineered shipment features.
    """
    df = shipments.copy()
    
    # Ensure datetime columns are parsed
    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
    df['scheduled_delivery'] = pd.to_datetime(df['scheduled_delivery'])
    
    # Time-based features
    df['departure_hour'] = df['scheduled_departure'].dt.hour
    df['departure_day_of_week'] = df['scheduled_departure'].dt.dayofweek
    df['departure_month'] = df['scheduled_departure'].dt.month
    df['is_weekend'] = df['departure_day_of_week'] >= 5
    
    # Duration
    df['scheduled_duration_hrs'] = (df['scheduled_delivery'] - df['scheduled_departure']).dt.total_seconds() / 3600.0
    
    # Weight class
    bins = [-np.inf, 500, 5000, np.inf]
    labels = ['light', 'medium', 'heavy']
    df['weight_class'] = pd.cut(df['weight_kg'], bins=bins, labels=labels).astype(str)
    
    # Categorical encoding
    le = LabelEncoder()
    if 'vehicle_type' in df.columns:
        df['vehicle_type_encoded'] = le.fit_transform(df['vehicle_type'].astype(str))
    else:
        df['vehicle_type_encoded'] = 0
        
    return df

def calculate_num_stops(group):
    """
    Calculates number of times speed < 5 for > 10 mins.
    
    Args:
        group (pd.DataFrame): Telemetry records for a single shipment.
        
    Returns:
        int: Number of stops.
    """
    if 'speed' not in group.columns or 'timestamp' not in group.columns:
         return 0
    
    group = group.sort_values('timestamp')
    group['timestamp'] = pd.to_datetime(group['timestamp'])
    
    is_stopped = group['speed'] < 5
    
    # Identify contiguous blocks of stopped time
    block_id = (is_stopped != is_stopped.shift()).cumsum()
    stopped_blocks = group[is_stopped].groupby(block_id)
    
    num_stops = 0
    for _, block in stopped_blocks:
        if len(block) > 1:
            duration = (block['timestamp'].iloc[-1] - block['timestamp'].iloc[0]).total_seconds() / 60.0
            if duration > 10:
                num_stops += 1
                
    return num_stops

def engineer_telemetry_features(telemetry):
    """
    Aggregates telemetry data into shipment-level features.
    
    Args:
        telemetry (pd.DataFrame): The raw telemetry dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with aggregated telemetry features.
    """
    telemetry['timestamp'] = pd.to_datetime(telemetry['timestamp'])
    
    # Ensure numeric columns
    for col in ['speed', 'fuel_level', 'odometer', 'lat', 'lon']:
        if col in telemetry.columns:
             telemetry[col] = pd.to_numeric(telemetry[col], errors='coerce')
    
    grouped = telemetry.groupby('shipment_id')
    features = pd.DataFrame(index=grouped.groups.keys())
    features.index.name = 'shipment_id'
    
    if 'speed' in telemetry.columns:
        features['avg_speed_kmh'] = grouped['speed'].mean()
        features['min_speed_kmh'] = grouped['speed'].min()
        features['max_speed_kmh'] = grouped['speed'].max()
        features['speed_variance'] = grouped['speed'].var().fillna(0)
    else:
        features[['avg_speed_kmh', 'min_speed_kmh', 'max_speed_kmh', 'speed_variance']] = 0
        
    features['num_stops'] = grouped.apply(calculate_num_stops)
    
    if 'fuel_level' in telemetry.columns:
        features['avg_fuel_level_pct'] = grouped['fuel_level'].mean()
        
        # Calculate fuel drop rate (fuel consumed per km)
        def drop_rate(g):
            if 'odometer' in g.columns and len(g) > 1:
                g = g.sort_values('timestamp')
                distance = g['odometer'].iloc[-1] - g['odometer'].iloc[0]
                fuel_used = g['fuel_level'].iloc[0] - g['fuel_level'].iloc[-1]
                if distance > 0:
                    return fuel_used / distance
            return np.nan
        features['fuel_drop_rate'] = grouped.apply(drop_rate)
    else:
        features['avg_fuel_level_pct'] = 0
        features['fuel_drop_rate'] = 0
        
    if 'engine_status' in telemetry.columns:
        features['pct_time_idle'] = grouped.apply(lambda g: (g['engine_status'] == 'idle').mean() * 100)
    else:
        features['pct_time_idle'] = 0
        
    # Route deviation placeholder: compute max deviation from planned route
    if 'deviation' in telemetry.columns:
        features['route_deviation_km'] = grouped['deviation'].max()
    else:
        features['route_deviation_km'] = 0
        
    return features.reset_index()

def engineer_disruption_features(shipments, disruptions):
    """
    Extracts features from disruptions overlapping with shipments.
    
    Args:
        shipments (pd.DataFrame): Shipments dataframe.
        disruptions (pd.DataFrame): Disruptions dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with disruption features per shipment.
    """
    features = []
    
    shipments['scheduled_departure'] = pd.to_datetime(shipments['scheduled_departure'])
    # Use scheduled_delivery if actual_delivery is missing for overlap calculation
    shipments['actual_delivery_coalesced'] = pd.to_datetime(shipments.get('actual_delivery', shipments['scheduled_delivery'])).fillna(shipments['scheduled_delivery'])
    
    disruptions['start_time'] = pd.to_datetime(disruptions['start_time'])
    disruptions['end_time'] = pd.to_datetime(disruptions['end_time'])
    
    for idx, shipment in shipments.iterrows():
        route_id = shipment.get('route_id')
        s_start = shipment['scheduled_departure']
        s_end = shipment['actual_delivery_coalesced']
        
        if pd.isna(route_id):
            features.append({
                'shipment_id': shipment['shipment_id'],
                'num_disruptions_on_route': 0,
                'max_disruption_severity': 0,
                'is_weather_disruption': False,
                'is_congestion_disruption': False,
                'disruption_overlap_hours': 0.0
            })
            continue
            
        route_disruptions = disruptions[disruptions['route_id'] == route_id]
        active = route_disruptions[(route_disruptions['start_time'] <= s_end) & (route_disruptions['end_time'] >= s_start)]
        
        if len(active) == 0:
            features.append({
                'shipment_id': shipment['shipment_id'],
                'num_disruptions_on_route': 0,
                'max_disruption_severity': 0,
                'is_weather_disruption': False,
                'is_congestion_disruption': False,
                'disruption_overlap_hours': 0.0
            })
            continue
            
        num_disruptions = len(active)
        max_severity = active['severity'].max() if 'severity' in active.columns else 0
        is_weather = 'weather' in active['type'].str.lower().values if 'type' in active.columns else False
        is_congestion = 'congestion' in active['type'].str.lower().values if 'type' in active.columns else False
        
        overlap_hours = 0
        for _, d in active.iterrows():
            overlap_start = max(s_start, d['start_time'])
            overlap_end = min(s_end, d['end_time'])
            overlap = (overlap_end - overlap_start).total_seconds() / 3600.0
            overlap_hours += max(0, overlap)
            
        features.append({
            'shipment_id': shipment['shipment_id'],
            'num_disruptions_on_route': num_disruptions,
            'max_disruption_severity': max_severity,
            'is_weather_disruption': is_weather,
            'is_congestion_disruption': is_congestion,
            'disruption_overlap_hours': overlap_hours
        })
        
    return pd.DataFrame(features)

def create_targets(shipments):
    """
    Creates the target variables for delay prediction.
    
    Args:
        shipments (pd.DataFrame): Shipments dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with target features per shipment.
    """
    df = shipments.copy()
    df['scheduled_delivery'] = pd.to_datetime(df['scheduled_delivery'])
    df['actual_delivery'] = pd.to_datetime(df['actual_delivery'])
    
    # Delay in minutes
    df['delay_minutes'] = (df['actual_delivery'] - df['scheduled_delivery']).dt.total_seconds() / 60.0
    
    # is_delayed (1 if delay > 60 min)
    df['is_delayed'] = (df['delay_minutes'] > 60).astype(int)
    
    return df[['shipment_id', 'delay_minutes', 'is_delayed']]

def handle_missing_values(df):
    """
    Imputes missing values using median for numerics and mode for categoricals.
    
    Args:
        df (pd.DataFrame): The combined features dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with imputed missing values.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    numeric_cols = [c for c in numeric_cols if c not in ['shipment_id', 'is_delayed', 'delay_minutes']]
    categorical_cols = [c for c in categorical_cols if c not in ['shipment_id']]
    
    if numeric_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        
    return df

def scale_features(df, model_dir='ml/models'):
    """
    Scales continuous features using StandardScaler and saves the scaler.
    
    Args:
        df (pd.DataFrame): Dataframe with features to scale.
        model_dir (str): Directory to save the fitted scaler.
        
    Returns:
        pd.DataFrame: Dataframe with scaled continuous features.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Define continuous columns (excluding targets, ids, booleans, categorical codes)
    exclude_cols = ['shipment_id', 'is_delayed', 'delay_minutes', 'departure_hour', 
                    'departure_day_of_week', 'departure_month', 'is_weekend', 
                    'weight_class', 'vehicle_type_encoded', 'is_weather_disruption', 
                    'is_congestion_disruption']
                    
    continuous_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    if continuous_cols:
        scaler = StandardScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
        
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
    return df

def main():
    """Main function to orchestrate the feature engineering pipeline."""
    # Assuming execution from project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Please ensure data is present.")
        return
        
    try:
        shipments, telemetry, disruptions = load_data(data_dir)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print("Engineering shipment features...")
    shipment_features = engineer_shipment_features(shipments)
    
    print("Engineering telemetry features...")
    telemetry_features = engineer_telemetry_features(telemetry)
    
    print("Engineering disruption features...")
    disruption_features = engineer_disruption_features(shipments, disruptions)
    
    print("Creating targets...")
    targets = create_targets(shipments)
    
    print("Merging features...")
    features_df = shipment_features.merge(telemetry_features, on='shipment_id', how='left')
    features_df = features_df.merge(disruption_features, on='shipment_id', how='left')
    features_df = features_df.merge(targets, on='shipment_id', how='left')
    
    print("Handling missing values...")
    features_df = handle_missing_values(features_df)
    
    print("Scaling continuous features...")
    features_df = scale_features(features_df, model_dir)
    
    # Drop original non-feature columns that aren't needed
    cols_to_drop = ['scheduled_departure', 'scheduled_delivery', 'actual_delivery', 
                    'actual_delivery_coalesced', 'vehicle_type', 'weight_kg', 'route_id']
    features_df = features_df.drop(columns=[c for c in cols_to_drop if c in features_df.columns])
    
    print("\n--- Class Balance (is_delayed) ---")
    if 'is_delayed' in features_df.columns:
        class_counts = features_df['is_delayed'].value_counts()
        print(class_counts)
        print(f"Delayed percentage: {(class_counts.get(1, 0) / len(features_df) * 100):.2f}%")
    
    print("\n--- Top 10 Feature Correlations with 'is_delayed' ---")
    if 'is_delayed' in features_df.columns:
        numeric_df = features_df.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()['is_delayed'].drop(['is_delayed', 'delay_minutes'], errors='ignore').abs().sort_values(ascending=False)
        print(correlations.head(10))
    
    output_path = os.path.join(data_dir, 'features.csv')
    features_df.to_csv(output_path, index=False)
    print(f"\nFeatures successfully saved to {output_path}")

if __name__ == "__main__":
    main()
