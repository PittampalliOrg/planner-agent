"""
Dapr Configuration and Secrets Provider

Provides a unified interface for accessing configuration values and secrets
through Dapr's Configuration and Secrets building blocks, with automatic
fallback to environment variables when Dapr is unavailable.
"""

from .provider import (
    initialize_config_and_secrets,
    get_config,
    get_secret_value,
    is_feature_enabled,
    is_dapr_enabled,
    is_initialized,
    refresh_configuration,
    refresh_secrets,
)

__all__ = [
    "initialize_config_and_secrets",
    "get_config",
    "get_secret_value",
    "is_feature_enabled",
    "is_dapr_enabled",
    "is_initialized",
    "refresh_configuration",
    "refresh_secrets",
]
