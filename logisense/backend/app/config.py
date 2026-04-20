"""
Configuration settings for LogiSense backend.
Uses Pydantic BaseSettings to load environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables or .env file.
    """
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LogiSense API"

    # Database Settings
    DATABASE_URL: str = "postgresql+asyncpg://logisense_user:logisense_password@localhost:5432/logisense_db"

    # Redis Settings
    REDIS_URL: str = "redis://localhost:6379/0"

    # Kafka Settings
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TELEMETRY_TOPIC: str = "telemetry_updates"
    KAFKA_DISRUPTION_TOPIC: str = "route_disruptions"

    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")


# Global settings instance to be imported by other modules
settings = Settings()
