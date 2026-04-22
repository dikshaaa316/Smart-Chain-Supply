"""
Synthetic Data Generator for LogiSense.
Generates realistic logistics data including shipments, telemetry, and disruptions.
"""

import uuid
import random
import datetime
import math
import logging
import os
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed list of 20 Indian cities and their approximate coordinates
CITIES = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Coimbatore": (11.0168, 76.9558),
    "Kochi": (9.9312, 76.2673),
    "Patna": (25.5941, 85.1376),
    "Bhubaneswar": (20.2961, 85.8245),
    "Chandigarh": (30.7333, 76.7794),
    "Visakhapatnam": (17.6868, 83.2185)
}

CITY_NAMES = list(CITIES.keys())
DISRUPTION_TYPES = ["weather", "accident", "congestion", "strike", "road_closure", "mechanical"]

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two lat/lon points in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_disruptions(num_disruptions: int = 2000) -> pd.DataFrame:
    """Generate synthetic route disruptions."""
    np.random.seed(42)
    now = datetime.datetime.now()
    
    data = []
    for _ in range(num_disruptions):
        route_id = f"R{np.random.randint(1, 201):03d}"
        dtype = np.random.choice(DISRUPTION_TYPES)
        severity = int(np.random.randint(1, 6))
        
        # Start time over last 180 days
        days_ago = np.random.randint(0, 180)
        hours_ago = np.random.randint(0, 24)
        start_time = now - datetime.timedelta(days=int(days_ago), hours=int(hours_ago))
        end_time = start_time + datetime.timedelta(hours=int(np.random.randint(1, 72)))
        
        lat = np.random.uniform(8.0, 37.0)
        lon = np.random.uniform(68.0, 97.0)
        
        data.append({
            "disruption_id": str(uuid.uuid4()),
            "route_id": route_id,
            "disruption_type": dtype,
            "severity": severity,
            "start_time": start_time,
            "end_time": end_time,
            "affected_radius_km": round(np.random.uniform(5.0, 50.0), 2),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "resolved": end_time < now
        })
    return pd.DataFrame(data)

def generate_shipments(num_shipments: int = 10000, disruptions_df: pd.DataFrame = None) -> pd.DataFrame:
    """Generate synthetic shipment data."""
    np.random.seed(42)
    now = datetime.datetime.now()
    
    # Pre-calculate severe route disruptions to simulate correlations
    severe_routes = set()
    if disruptions_df is not None:
        severe_df = disruptions_df[disruptions_df["severity"] >= 3]
        severe_routes = set(severe_df["route_id"].unique())

    data = []
    for _ in range(num_shipments):
        origin = np.random.choice(CITY_NAMES)
        dest = np.random.choice([c for c in CITY_NAMES if c != origin])
        
        route_id = f"R{np.random.randint(1, 201):03d}"
        
        # Calculate distance
        lat1, lon1 = CITIES[origin]
        lat2, lon2 = CITIES[dest]
        dist_km = haversine(lat1, lon1, lat2, lon2)
        
        scheduled_departure = now - datetime.timedelta(
            days=int(np.random.randint(0, 180)), 
            hours=int(np.random.randint(0, 24))
        )
        
        # Base delay logic
        has_severe = route_id in severe_routes
        max_dep_delay = 8 if has_severe else 4
        dep_delay_hrs = np.random.uniform(0, max_dep_delay)
        actual_departure = scheduled_departure + datetime.timedelta(hours=float(dep_delay_hrs))
        
        # Duration based on distance (assume avg 60 km/h)
        base_duration_hrs = dist_km / 60.0
        scheduled_delivery = scheduled_departure + datetime.timedelta(hours=float(base_duration_hrs))
        
        # Actual delivery (70% filled)
        if np.random.random() < 0.7:
            # Delivered or Delayed
            del_delay_hrs = np.random.uniform(-2, 12 if has_severe else 4)
            actual_delivery = scheduled_delivery + datetime.timedelta(hours=float(del_delay_hrs))
            
            if actual_delivery > scheduled_delivery + datetime.timedelta(hours=2):
                status = "delayed"
            else:
                status = "delivered"
        else:
            actual_delivery = None
            status = np.random.choice(["in_transit", "cancelled"], p=[0.9, 0.1])
            
        data.append({
            "shipment_id": str(uuid.uuid4()),
            "origin_city": origin,
            "destination_city": dest,
            "carrier_id": f"C{np.random.randint(1, 51):03d}",
            "vehicle_type": np.random.choice(["truck", "rail", "air"], p=[0.7, 0.2, 0.1]),
            "weight_kg": round(float(np.random.uniform(100, 20000)), 2),
            "scheduled_departure": scheduled_departure,
            "actual_departure": actual_departure,
            "scheduled_delivery": scheduled_delivery,
            "actual_delivery": actual_delivery,
            "status": status,
            "route_id": route_id,
            "_dist_km": dist_km, # Temp field for telemetry correlation
            "_has_severe": has_severe # Temp field for telemetry correlation
        })
        
    return pd.DataFrame(data)

def generate_telemetry(shipments_df: pd.DataFrame, points_per_shipment: int = 50) -> pd.DataFrame:
    """Generate synthetic telemetry data for shipments."""
    np.random.seed(42)
    data = []
    
    for _, row in shipments_df.iterrows():
        sid = row["shipment_id"]
        lat1, lon1 = CITIES[row["origin_city"]]
        lat2, lon2 = CITIES[row["destination_city"]]
        dist_km = row["_dist_km"]
        has_severe = row["_has_severe"]
        
        start_time = row["actual_departure"]
        if pd.notnull(row["actual_delivery"]):
            end_time = row["actual_delivery"]
        else:
            end_time = start_time + datetime.timedelta(hours=float(dist_km/60.0))
            
        time_step = (end_time - start_time) / max(1, (points_per_shipment - 1))
        
        # Fuel drops faster on longer routes
        fuel_drop_rate = np.random.uniform(1.0, 2.5) * (dist_km / 1000.0)
        current_fuel = 100.0
        
        for i in range(points_per_shipment):
            current_time = start_time + (time_step * i)
            progress = i / max(1, (points_per_shipment - 1))
            
            # Linear interpolation with noise
            lat = lat1 + (lat2 - lat1) * progress + np.random.normal(0, 0.05)
            lon = lon1 + (lon2 - lon1) * progress + np.random.normal(0, 0.05)
            
            # Speeds drop near disruption (simulate as midpoint slowdown)
            base_speed = np.random.uniform(40, 80)
            if has_severe and 0.4 < progress < 0.6:
                base_speed = np.random.uniform(0, 20)
                
            # Occasional stops
            if np.random.random() < 0.05:
                base_speed = 0.0
                
            # Fuel logic
            current_fuel = max(0.0, current_fuel - fuel_drop_rate)
            if current_fuel < 10.0 and progress < 0.9:
                current_fuel = 100.0 # Refuel
                base_speed = 0.0
                
            engine_status = "off" if base_speed == 0 and np.random.random() < 0.3 else ("idle" if base_speed == 0 else "running")
            
            data.append({
                "telemetry_id": str(uuid.uuid4()),
                "shipment_id": sid,
                "timestamp": current_time,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "speed_kmh": round(base_speed, 2),
                "fuel_level_pct": round(current_fuel, 2),
                "temperature_celsius": round(float(np.random.uniform(20, 40)), 1),
                "engine_status": engine_status,
                "driver_id": f"D{np.random.randint(1, 501):03d}"
            })
            
    return pd.DataFrame(data)

def main():
    """Main execution function."""
    logger.info("Starting synthetic data generation...")
    
    # Generate data
    disruptions_df = generate_disruptions(2000)
    logger.info(f"Generated {len(disruptions_df)} disruptions.")
    
    shipments_df = generate_shipments(10000, disruptions_df)
    logger.info(f"Generated {len(shipments_df)} shipments.")
    
    telemetry_df = generate_telemetry(shipments_df, 50)
    logger.info(f"Generated {len(telemetry_df)} telemetry readings.")
    
    # Clean up temporary columns
    shipments_df = shipments_df.drop(columns=["_dist_km", "_has_severe"])
    
    # Save to CSV
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    disruptions_path = os.path.join(output_dir, "disruptions.csv")
    shipments_path = os.path.join(output_dir, "shipments.csv")
    telemetry_path = os.path.join(output_dir, "telemetry.csv")
    
    disruptions_df.to_csv(disruptions_path, index=False)
    shipments_df.to_csv(shipments_path, index=False)
    telemetry_df.to_csv(telemetry_path, index=False)
    
    logger.info(f"Data generation complete. Files saved to {output_dir}")
    print(f"Row counts -> Shipments: {len(shipments_df)}, Telemetry: {len(telemetry_df)}, Disruptions: {len(disruptions_df)}")

if __name__ == "__main__":
    main()
