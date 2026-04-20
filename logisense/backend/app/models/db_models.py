"""
SQLAlchemy ORM Models for LogiSense.
Using SQLAlchemy 2.0 mapped_column declarative style.
"""

from typing import Optional, List
from datetime import datetime
import uuid

from sqlalchemy import (
    String, 
    Float, 
    Integer, 
    Boolean, 
    DateTime, 
    ForeignKey, 
    text,
    CheckConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

class Base(DeclarativeBase):
    pass

class Route(Base):
    __tablename__ = "routes"

    route_id: Mapped[str] = mapped_column(String, primary_key=True)
    origin_city: Mapped[Optional[str]] = mapped_column(String)
    destination_city: Mapped[Optional[str]] = mapped_column(String)
    distance_km: Mapped[Optional[float]] = mapped_column(Float)
    estimated_duration_hrs: Mapped[Optional[float]] = mapped_column(Float)
    waypoints: Mapped[Optional[dict]] = mapped_column(JSONB)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")

    # Relationships
    shipments: Mapped[List["Shipment"]] = relationship(back_populates="route")
    disruptions: Mapped[List["Disruption"]] = relationship(back_populates="route")

class Shipment(Base):
    __tablename__ = "shipments"

    shipment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    route_id: Mapped[Optional[str]] = mapped_column(ForeignKey("routes.route_id"))
    carrier_id: Mapped[Optional[str]] = mapped_column(String)
    vehicle_type: Mapped[Optional[str]] = mapped_column(String)
    weight_kg: Mapped[Optional[float]] = mapped_column(Float)
    scheduled_departure: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    actual_departure: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    scheduled_delivery: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    actual_delivery: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, default="scheduled", server_default="scheduled")
    current_lat: Mapped[Optional[float]] = mapped_column(Float)
    current_lon: Mapped[Optional[float]] = mapped_column(Float)
    predicted_delay_minutes: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    risk_level: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))

    __table_args__ = (
        CheckConstraint("vehicle_type IN ('truck', 'rail', 'air')", name="shipments_vehicle_type_check"),
        CheckConstraint("risk_level BETWEEN 1 AND 5", name="shipments_risk_level_check"),
    )

    # Relationships
    route: Mapped[Optional["Route"]] = relationship(back_populates="shipments")
    telemetry_readings: Mapped[List["Telemetry"]] = relationship(back_populates="shipment")
    alerts: Mapped[List["Alert"]] = relationship(back_populates="shipment")

class Telemetry(Base):
    __tablename__ = "telemetry"

    # For TimescaleDB hypertables, if a PK is defined, the partition column (recorded_at) MUST be part of it.
    telemetry_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    shipment_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("shipments.shipment_id"))
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    speed_kmh: Mapped[Optional[float]] = mapped_column(Float)
    fuel_level_pct: Mapped[Optional[float]] = mapped_column(Float)
    engine_status: Mapped[Optional[str]] = mapped_column(String)
    driver_id: Mapped[Optional[str]] = mapped_column(String)

    # Relationships
    shipment: Mapped[Optional["Shipment"]] = relationship(back_populates="telemetry_readings")

class Disruption(Base):
    __tablename__ = "disruptions"

    disruption_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    route_id: Mapped[Optional[str]] = mapped_column(ForeignKey("routes.route_id"))
    disruption_type: Mapped[Optional[str]] = mapped_column(String)
    severity: Mapped[Optional[int]] = mapped_column(Integer)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    affected_radius_km: Mapped[Optional[float]] = mapped_column(Float)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    __table_args__ = (
        CheckConstraint("severity BETWEEN 1 AND 5", name="disruptions_severity_check"),
    )

    # Relationships
    route: Mapped[Optional["Route"]] = relationship(back_populates="disruptions")

class RouteAlternative(Base):
    __tablename__ = "route_alternatives"

    alt_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    original_route_id: Mapped[Optional[str]] = mapped_column(ForeignKey("routes.route_id"))
    alternative_route_id: Mapped[Optional[str]] = mapped_column(ForeignKey("routes.route_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))
    estimated_time_saving_mins: Mapped[Optional[int]] = mapped_column(Integer)
    extra_distance_km: Mapped[Optional[float]] = mapped_column(Float)
    reason: Mapped[Optional[str]] = mapped_column(String)

class Alert(Base):
    __tablename__ = "alerts"

    alert_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    shipment_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("shipments.shipment_id"))
    alert_type: Mapped[Optional[str]] = mapped_column(String)
    severity: Mapped[Optional[int]] = mapped_column(Integer)
    message: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    # Relationships
    shipment: Mapped[Optional["Shipment"]] = relationship(back_populates="alerts")
