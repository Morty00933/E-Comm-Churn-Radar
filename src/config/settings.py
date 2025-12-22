from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    
    @property
    def host(self) -> str:
        return self.url.split("://")[1].split(":")[0] if "://" in self.url else "localhost"
    
    @property
    def port(self) -> int:
        try:
            return int(self.url.split(":")[-1].split("/")[0])
        except (ValueError, IndexError):
            return 6379


class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_")
    
    tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    model_name: str = Field(default="churn-model", alias="MLFLOW_MODEL_NAME")
    model_stage: str = Field(default="Production", alias="MLFLOW_MODEL_STAGE")


class RateLimitSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")
    
    enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    requests: int = Field(default=100, alias="RATE_LIMIT_REQUESTS")
    window_sec: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SEC")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    app_name: str = Field(default="Churn Radar")
    app_version: str = Field(default="1.0.0")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        alias="ENVIRONMENT"
    )
    debug: bool = Field(default=False, alias="DEBUG")
    
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_key: str = Field(default="dev-api-key", alias="API_KEY")
    
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=False, alias="LOG_JSON")
    
    redis: RedisSettings = Field(default_factory=RedisSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


settings = get_settings()
