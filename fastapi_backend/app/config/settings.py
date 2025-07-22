"""
Application Settings and Configuration
"""

import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    api_title: str = "Survey Analysis API"
    api_version: str = "1.0.0"
    api_description: str = "Statistical analysis system for course evaluation surveys"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    
    # Database Configuration
    database_url: str = "sqlite:///./survey_analysis.db"
    database_echo: bool = False
    
    # LLM Configuration
    llm_provider: str = "openai"  # openai, anthropic, local
    llm_model: str = "gpt-4.1-2025-04-14"
    llm_api_key: Optional[str] = None
    llm_max_tokens: int = 4000
    llm_temperature: float = 0.3
    
    # Analysis Configuration
    min_sample_size: int = 1
    confidence_level: float = 0.95
    max_clusters: int = 6
    statistical_significance: float = 0.05
    
    # Security Configuration
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS Configuration
    cors_origins: list = ["http://localhost:3000", "http://localhost:8080"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]
    
    # File Upload Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: list = [".json", ".csv", ".xlsx"]
    
    # Cache Configuration
    redis_url: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()