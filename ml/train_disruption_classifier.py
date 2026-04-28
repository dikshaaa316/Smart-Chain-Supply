import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

FEATURE_COLS = [
    'departure_hour', 'departure_day_of_week', 'departure_month', 'is_weekend',
    'scheduled_duration_hrs', 'weight_class', 'vehicle_type_encoded',
    'avg_speed_kmh', 'min_speed_kmh', 'speed_variance', 'num_stops', 'avg_fuel_level_pct',
    'fuel_drop_rate', 'pct_time_idle', 'route_deviation_km',
    'num_disruptions_on_route', 'max_disruption_severity', 'is_weather_disruption',
    'is_congestion_disruption', 'disruption_overlap_hours'
]

def load_and_preprocess_data(data_path):
    """Loads features and drops rows with null targets."""
    df = pd.read_csv(data_path)
    
    # 1. Drop rows where is_delayed is null
    df = df.dropna(subset=['is_delayed'])
    
    # Label encode weight_class if it's not already numeric
    if 'weight_class' in df.columns and df['weight_class'].dtype == 'object':
        le = LabelEncoder()
        df['weight_class'] = le.fit_transform(df['weight_class'].astype(str))
        
    # Ensure all required features are present
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"Warning: Missing column {col}. Filling with 0.")
            df[col] = 0
            
    # Convert booleans to integers
    for col in FEATURE_COLS:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
            
    X = df[FEATURE_COLS]
    y = df['is_delayed'].astype(int)
    return X, y

def train_and_evaluate_models(X, y, model_dir):
    """Trains 3 models, prints evaluation metrics, and saves the best model."""
    # 3. Train/test split: 80/20 stratified on is_delayed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 7. Handle class imbalance for XGBoost
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # 4. Define models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBClassifier': XGBClassifier(
            n_estimators=300, 
            max_depth=6, 
            learning_rate=0.05, 
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42
        ),
        'LGBMClassifier': LGBMClassifier(
            n_estimators=300, 
            max_depth=6, 
            learning_rate=0.05,
            random_state=42
        )
    }
    
    best_model_name = None
    best_roc_auc = -1
    best_model = None
    
    # 5. Train and evaluate each model
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        # 6. For XGBoost, print top 15 feature importances
        if name == 'XGBClassifier':
            print("\nTop 15 Feature Importances (XGBoost):")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(min(15, len(FEATURE_COLS))):
                col_name = FEATURE_COLS[indices[i]]
                imp = importances[indices[i]]
                bar = "#" * int(imp * 50)  # visual bar length scaling
                print(f"{col_name[:25]:>25} | {imp:.4f} | {bar}")
                
        # Track the best model by ROC-AUC
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
    
    # 8. Save the best model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
    model_path = os.path.join(model_dir, 'disruption_classifier.pkl')
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")
    
    # 9. Save model metadata
    metadata = {
        "model_type": best_model_name,
        "feature_cols": FEATURE_COLS,
        "roc_auc": float(best_roc_auc),
        "threshold": 0.5,
        "trained_at": datetime.utcnow().isoformat() + "Z"
    }
    
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {metadata_path}")
    
def predict_disruption(feature_dict, model_dir=None):
    """
    Predicts disruption based on a dictionary of features.
    
    Args:
        feature_dict (dict): Dictionary mapping feature names to values.
        model_dir (str, optional): Directory containing the models. Defaults to ml/models.
        
    Returns:
        dict: Prediction results including is_delayed, delay_probability, and risk_level.
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
    model_path = os.path.join(model_dir, 'disruption_classifier.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model = joblib.load(model_path)
    df = pd.DataFrame([feature_dict])
    
    # Fill missing features with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
            
    # Quick fallback for categorical weight_class
    if 'weight_class' in df.columns and df['weight_class'].dtype == 'object':
        weight_map = {'light': 0, 'medium': 1, 'heavy': 2}
        df['weight_class'] = df['weight_class'].map(weight_map).fillna(0)
            
    # Load and apply scaler if it exists
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            # Find which columns are numeric to scale (mirroring feature_engineering.py)
            exclude_cols = ['shipment_id', 'is_delayed', 'delay_minutes', 'departure_hour', 
                            'departure_day_of_week', 'departure_month', 'is_weekend', 
                            'weight_class', 'vehicle_type_encoded', 'is_weather_disruption', 
                            'is_congestion_disruption']
            continuous_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
            
            # Apply scaling
            if len(continuous_cols) > 0 and hasattr(scaler, 'n_features_in_'):
                if scaler.n_features_in_ == len(continuous_cols):
                    df[continuous_cols] = scaler.transform(df[continuous_cols])
        except Exception as e:
            print(f"Warning: Could not apply scaler during inference: {e}")
            
    X = df[FEATURE_COLS]
    
    # Extract probability of class 1
    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else float(model.predict(X)[0])
    is_delayed = bool(prob >= 0.5)
    
    # Determine risk level (1-5)
    if prob <= 0.2:
        risk_level = 1
    elif prob <= 0.4:
        risk_level = 2
    elif prob <= 0.6:
        risk_level = 3
    elif prob <= 0.8:
        risk_level = 4
    else:
        risk_level = 5
        
    return {
        "is_delayed": is_delayed,
        "delay_probability": float(prob),
        "risk_level": risk_level
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'features.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please run feature_engineering.py to generate it.")
        return
        
    print(f"Loading data from {data_path}...")
    try:
        X, y = load_and_preprocess_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Starting model training and evaluation with {len(X)} samples...")
    train_and_evaluate_models(X, y, model_dir=model_dir)

if __name__ == '__main__':
    main()
