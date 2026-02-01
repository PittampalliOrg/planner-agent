"""
Dapr HTTP Client for Configuration and Secrets APIs

Provides low-level HTTP access to Dapr's Configuration and Secrets building blocks.
"""

import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DAPR_HTTP_PORT = os.environ.get("DAPR_HTTP_PORT", "3500")
DAPR_HOST = os.environ.get("DAPR_HOST", "localhost")


def dapr_url(path: str) -> str:
    """Build a Dapr sidecar URL."""
    return f"http://{DAPR_HOST}:{DAPR_HTTP_PORT}{path}"


async def is_available(timeout: float = 3.0) -> bool:
    """Check if Dapr sidecar is available and healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                dapr_url("/v1.0/healthz"),
                timeout=timeout,
            )
            return response.status_code == 200
    except Exception:
        return False


async def get_configuration(
    store_name: str,
    keys: list[str],
    label: Optional[str] = None,
    metadata: Optional[dict[str, str]] = None,
    timeout: float = 5.0,
) -> dict[str, dict[str, Any]]:
    """
    Get configuration values from a Dapr configuration store.

    Args:
        store_name: Name of the configuration store component (e.g., "azureappconfig")
        keys: List of configuration keys to retrieve
        label: Optional label to filter configuration items
        metadata: Optional additional metadata parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary of key to ConfigurationItem (with 'value', 'version', 'metadata' fields)

    Example:
        config = await get_configuration("azureappconfig", ["OPENAI_MODEL"], label="planner-agent")
        print(config["OPENAI_MODEL"]["value"])  # "gpt-4o"
    """
    try:
        url = dapr_url(f"/v1.0/configuration/{store_name}")
        params: dict[str, Any] = {}

        # Add keys as multiple key= params
        if keys:
            params["key"] = keys

        # Add label if provided
        if label:
            params["metadata.label"] = label

        # Add any additional metadata
        if metadata:
            for key, value in metadata.items():
                params[f"metadata.{key}"] = value

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=timeout)

            if response.status_code != 200:
                raise Exception(f"Configuration get failed: {response.status_code}")

            return response.json()

    except Exception as e:
        logger.error(f"[DaprClient] Failed to get configuration from {store_name}: {e}")
        raise


async def get_secret(
    store_name: str,
    secret_name: str,
    metadata: Optional[dict[str, str]] = None,
    timeout: float = 5.0,
) -> str:
    """
    Get a single secret from a Dapr secrets store.

    Args:
        store_name: Name of the secrets store component (e.g., "azurekeyvault")
        secret_name: Name of the secret to retrieve
        metadata: Optional metadata for the request
        timeout: Request timeout in seconds

    Returns:
        The secret value as a string

    Example:
        api_key = await get_secret("azurekeyvault", "OPENAI-API-KEY")
    """
    try:
        url = dapr_url(f"/v1.0/secrets/{store_name}/{secret_name}")
        params: dict[str, str] = {}

        if metadata:
            for key, value in metadata.items():
                params[f"metadata.{key}"] = value

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=timeout)

            if response.status_code != 200:
                raise Exception(f"Secret get failed: {response.status_code}")

            data = response.json()
            # Dapr returns { secretName: secretValue }
            return data.get(secret_name, "")

    except Exception as e:
        logger.error(f"[DaprClient] Failed to get secret {secret_name} from {store_name}: {e}")
        raise


async def get_bulk_secrets(
    store_name: str,
    timeout: float = 10.0,
) -> dict[str, str]:
    """
    Get all secrets from a Dapr secrets store (bulk operation).

    Note: Not all secret stores support bulk operations.
    Azure Key Vault and Kubernetes secrets do support this.

    Args:
        store_name: Name of the secrets store component
        timeout: Request timeout in seconds

    Returns:
        Dictionary of secret name to secret value
    """
    try:
        url = dapr_url(f"/v1.0/secrets/{store_name}/bulk")

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=timeout)

            if response.status_code != 200:
                raise Exception(f"Bulk secrets get failed: {response.status_code}")

            data = response.json()
            # Dapr returns { secretName: { secretName: value } } for bulk
            flattened: dict[str, str] = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    # Get the first value from the inner dict
                    flattened[key] = next(iter(value.values()), "")
                else:
                    flattened[key] = str(value)

            return flattened

    except Exception as e:
        logger.error(f"[DaprClient] Failed to get bulk secrets from {store_name}: {e}")
        raise
