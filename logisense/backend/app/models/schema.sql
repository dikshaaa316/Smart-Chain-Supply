-- LogiSense PostgreSQL + TimescaleDB Schema

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Routes Table
CREATE TABLE IF NOT EXISTS routes (
    route_id VARCHAR PRIMARY KEY,
    origin_city VARCHAR,
    destination_city VARCHAR,
    distance_km FLOAT,
    estimated_duration_hrs FLOAT,
    waypoints JSONB,
    risk_score FLOAT DEFAULT 0
);

-- 2. Shipments Table
CREATE TABLE IF NOT EXISTS shipments (
    shipment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id VARCHAR REFERENCES routes(route_id),
    carrier_id VARCHAR,
    vehicle_type VARCHAR CHECK (vehicle_type IN ('truck','rail','air')),
    weight_kg FLOAT,
    scheduled_departure TIMESTAMPTZ,
    actual_departure TIMESTAMPTZ,
    scheduled_delivery TIMESTAMPTZ,
    actual_delivery TIMESTAMPTZ,
    status VARCHAR DEFAULT 'scheduled',
    current_lat FLOAT,
    current_lon FLOAT,
    predicted_delay_minutes INT DEFAULT 0,
    risk_level INT DEFAULT 1 CHECK (risk_level BETWEEN 1 AND 5),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Telemetry Table (Hypertable)
CREATE TABLE IF NOT EXISTS telemetry (
    telemetry_id UUID DEFAULT gen_random_uuid(),
    shipment_id UUID REFERENCES shipments(shipment_id),
    recorded_at TIMESTAMPTZ NOT NULL,
    latitude FLOAT,
    longitude FLOAT,
    speed_kmh FLOAT,
    fuel_level_pct FLOAT,
    engine_status VARCHAR,
    driver_id VARCHAR
);

-- Convert to hypertable partitioned by time
SELECT create_hypertable('telemetry', 'recorded_at', if_not_exists => TRUE);

-- 4. Disruptions Table
CREATE TABLE IF NOT EXISTS disruptions (
    disruption_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id VARCHAR REFERENCES routes(route_id),
    disruption_type VARCHAR,
    severity INT CHECK (severity BETWEEN 1 AND 5),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    latitude FLOAT,
    longitude FLOAT,
    affected_radius_km FLOAT,
    resolved BOOLEAN DEFAULT false
);

-- 5. Route Alternatives Table
CREATE TABLE IF NOT EXISTS route_alternatives (
    alt_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_route_id VARCHAR REFERENCES routes(route_id),
    alternative_route_id VARCHAR REFERENCES routes(route_id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    estimated_time_saving_mins INT,
    extra_distance_km FLOAT,
    reason TEXT
);

-- 6. Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    shipment_id UUID REFERENCES shipments(shipment_id),
    alert_type VARCHAR,
    severity INT,
    message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT false
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_shipments_status ON shipments(status);
CREATE INDEX IF NOT EXISTS idx_telemetry_shipment_id ON telemetry(shipment_id);
CREATE INDEX IF NOT EXISTS idx_disruptions_route_resolved ON disruptions(route_id, resolved);

-- View: active_shipments_risk_view
CREATE OR REPLACE VIEW active_shipments_risk_view AS
WITH latest_telemetry AS (
    SELECT DISTINCT ON (shipment_id)
        shipment_id,
        recorded_at,
        speed_kmh,
        fuel_level_pct
    FROM telemetry
    ORDER BY shipment_id, recorded_at DESC
),
route_disruptions AS (
    SELECT 
        route_id, 
        MAX(severity) AS max_severity
    FROM disruptions
    WHERE resolved = false
    GROUP BY route_id
)
SELECT 
    s.shipment_id,
    s.status,
    s.risk_level,
    s.predicted_delay_minutes,
    r.route_id,
    r.origin_city,
    r.destination_city,
    r.risk_score AS route_risk_score,
    lt.recorded_at AS last_telemetry_time,
    lt.speed_kmh,
    lt.fuel_level_pct,
    COALESCE(rd.max_severity, 0) AS max_active_disruption_severity
FROM shipments s
JOIN routes r ON s.route_id = r.route_id
LEFT JOIN latest_telemetry lt ON s.shipment_id = lt.shipment_id
LEFT JOIN route_disruptions rd ON s.route_id = rd.route_id
WHERE s.status NOT IN ('delivered', 'cancelled');
