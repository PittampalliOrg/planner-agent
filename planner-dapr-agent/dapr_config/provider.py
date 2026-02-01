"""
Dapr Configuration and Secrets Provider

Provides a unified interface for accessing configuration values and secrets
through Dapr's Configuration and Secrets building blocks.

Features:
- Caches configuration and secrets at startup
- Falls back to environment variables when Dapr is unavailable
- Supports dynamic configuration updates (when subscribed)
- Type-safe access to known configuration keys

Example:
    # Initialize at startup (in app.py)
    await initialize_config_and_secrets()

    # Use throughout the application
    api_key = get_secret_value("OPENAI_API_KEY")
    model = get_config("OPENAI_MODEL", "gpt-4o")
"""

import asyncio
import logging
import os
from typing import Optional

from . import client

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Default Dapr component names
# These can be overridden via environment variables for flexibility
CONFIG_STORE = os.environ.get("DAPR_CONFIG_STORE", "azureappconfig")
SECRET_STORE = os.environ.get("DAPR_SECRET_STORE", "azurekeyvault")

# Configuration keys to load from Dapr Configuration store (Azure App Configuration)
# These are non-sensitive values that benefit from dynamic updates
CONFIG_KEYS = [
    # Model settings
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    # Pub/sub settings
    "PUBSUB_NAME",
    "PUBSUB_TOPIC",
    # State store settings
    "WORKFLOW_INDEX_STORE",
    # Feature flags
    "DAPR_WORKFLOW_ENABLED",
    # Observability
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_SERVICE_NAME",
]

# Secret mappings from application env var names to Azure Key Vault secret names
# Format: { ENV_VAR_NAME: "AZURE-KEY-VAULT-SECRET-NAME" }
# Azure Key Vault uses hyphens in secret names, while our app uses underscores.
SECRET_MAPPINGS: dict[str, str] = {
    # AI Providers
    "OPENAI_API_KEY": "OPENAI-API-KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC-API-KEY",
}

# ============================================================================
# State
# ============================================================================

# Cache for configuration values
_config_cache: dict[str, str] = {}

# Cache for secret values
_secret_cache: dict[str, str] = {}

# Whether the provider has been initialized
_initialized = False

# Promise for in-progress initialization
_initialization_lock = asyncio.Lock()

# Whether Dapr is available
_dapr_available = False

# ============================================================================
# Initialization
# ============================================================================


async def initialize_config_and_secrets() -> None:
    """
    Initialize the configuration and secrets provider.

    This should be called during application startup.
    It will:
    1. Check if Dapr sidecar is available
    2. Load configuration from Dapr Configuration store (or fall back to env vars)
    3. Load secrets from Dapr Secrets store (or fall back to env vars)
    """
    global _initialized, _dapr_available

    async with _initialization_lock:
        if _initialized:
            return

        logger.info("[ConfigProvider] Initializing configuration and secrets...")

        # Check if Dapr sidecar is available
        try:
            _dapr_available = await client.is_available()
        except Exception:
            _dapr_available = False

        if not _dapr_available:
            logger.info("[ConfigProvider] Dapr sidecar not available, using environment variables")
            _load_from_environment()
            _initialized = True
            return

        logger.info("[ConfigProvider] Dapr sidecar available, loading from Dapr stores...")

        # Load configuration from Dapr
        try:
            await _load_configuration_from_dapr()
        except Exception as e:
            logger.warning(f"[ConfigProvider] Failed to load configuration from Dapr, falling back to env vars: {e}")
            _load_config_from_environment()

        # Load secrets from Dapr
        try:
            await _load_secrets_from_dapr()
        except Exception as e:
            logger.warning(f"[ConfigProvider] Failed to load secrets from Dapr, falling back to env vars: {e}")
            _load_secrets_from_environment()

        _initialized = True
        logger.info("[ConfigProvider] Initialization complete")


async def _load_configuration_from_dapr() -> None:
    """Load configuration values from Dapr Configuration store."""
    global _config_cache

    try:
        config = await client.get_configuration(
            CONFIG_STORE,
            CONFIG_KEYS,
            label="planner-agent",
        )

        for key in CONFIG_KEYS:
            item = config.get(key)
            if item and "value" in item:
                _config_cache[key] = item["value"]
            else:
                # Fall back to environment variable
                env_value = os.environ.get(key)
                if env_value is not None:
                    _config_cache[key] = env_value

        logger.info(f"[ConfigProvider] Loaded {len(_config_cache)} configuration values")

    except Exception as e:
        logger.error(f"[ConfigProvider] Error loading configuration: {e}")
        raise


async def _load_secrets_from_dapr() -> None:
    """Load secrets from Dapr Secrets store."""
    global _secret_cache

    for env_key, kv_name in SECRET_MAPPINGS.items():
        try:
            value = await client.get_secret(SECRET_STORE, kv_name)
            if value:
                _secret_cache[env_key] = value
        except Exception:
            # Fall back to environment variable for this specific secret
            env_value = os.environ.get(env_key)
            if env_value is not None:
                _secret_cache[env_key] = env_value
                logger.warning(f"[ConfigProvider] Using env var fallback for secret: {env_key}")

    logger.info(f"[ConfigProvider] Loaded {len(_secret_cache)} secrets")


def _load_from_environment() -> None:
    """Load all values from environment variables (fallback mode)."""
    _load_config_from_environment()
    _load_secrets_from_environment()


def _load_config_from_environment() -> None:
    """Load configuration values from environment variables."""
    global _config_cache

    for key in CONFIG_KEYS:
        value = os.environ.get(key)
        if value is not None:
            _config_cache[key] = value


def _load_secrets_from_environment() -> None:
    """Load secrets from environment variables."""
    global _secret_cache

    for env_key in SECRET_MAPPINGS.keys():
        value = os.environ.get(env_key)
        if value is not None:
            _secret_cache[env_key] = value


# ============================================================================
# Accessors
# ============================================================================


def get_config(key: str, default: Optional[str] = None) -> str:
    """
    Get a configuration value.

    Priority order:
    1. Dapr Configuration cache (if initialized from Dapr)
    2. Environment variable
    3. Default value

    Args:
        key: Configuration key
        default: Default value if not found

    Returns:
        The configuration value or default

    Example:
        model = get_config("OPENAI_MODEL", "gpt-4o")
    """
    # Check cache first (populated from Dapr or env vars)
    if key in _config_cache:
        return _config_cache[key]

    # Direct env var fallback (for values not in CONFIG_KEYS)
    env_value = os.environ.get(key)
    if env_value is not None:
        return env_value

    return default or ""


def get_secret_value(key: str) -> str:
    """
    Get a secret value.

    Priority order:
    1. Dapr Secrets cache (if initialized from Dapr)
    2. Environment variable

    Args:
        key: Secret key (using ENV_VAR naming convention)

    Returns:
        The secret value or empty string

    Example:
        api_key = get_secret_value("OPENAI_API_KEY")
    """
    # Check cache first
    if key in _secret_cache:
        return _secret_cache[key]

    # Direct env var fallback
    env_value = os.environ.get(key)
    if env_value is not None:
        return env_value

    return ""


def is_feature_enabled(key: str) -> bool:
    """
    Check if a feature flag is enabled.

    Args:
        key: Feature flag key

    Returns:
        True if the value is "true" or "1"

    Example:
        if is_feature_enabled("DAPR_WORKFLOW_ENABLED"):
            # Use Dapr workflow
            pass
    """
    value = get_config(key, "false")
    return value.lower() in ("true", "1", "yes")


def is_dapr_enabled() -> bool:
    """Check if Dapr building blocks are being used."""
    return _dapr_available and _initialized


def is_initialized() -> bool:
    """Check if the provider has been initialized."""
    return _initialized


# ============================================================================
# Dynamic Updates
# ============================================================================


async def refresh_configuration() -> None:
    """
    Refresh configuration from Dapr.

    Call this to manually refresh configuration values.
    Note: For automatic updates, use Dapr's subscription feature.
    """
    if not _dapr_available:
        logger.info("[ConfigProvider] Dapr not available, skipping refresh")
        return

    try:
        await _load_configuration_from_dapr()
        logger.info("[ConfigProvider] Configuration refreshed")
    except Exception as e:
        logger.error(f"[ConfigProvider] Failed to refresh configuration: {e}")


async def refresh_secrets() -> None:
    """
    Refresh secrets from Dapr.

    Call this after secret rotation to pick up new values.
    Note: Secrets should generally not be cached long-term.
    """
    if not _dapr_available:
        logger.info("[ConfigProvider] Dapr not available, skipping refresh")
        return

    try:
        await _load_secrets_from_dapr()
        logger.info("[ConfigProvider] Secrets refreshed")
    except Exception as e:
        logger.error(f"[ConfigProvider] Failed to refresh secrets: {e}")
