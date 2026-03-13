"""
ALCF Globus token resolution.

Tries three methods in order:

1. ``ALCF_ACCESS_TOKEN`` environment variable (fastest, good for CI).
2. ``globus_sdk`` (``pip install aure[alcf]``) — reuses cached tokens.
3. ``inference_auth_token.py get_access_token`` subprocess fallback.
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# Globus app constants – configurable via env vars, defaults mirror
# inference_auth_token.py from the ALCF inference-endpoints repo.
APP_NAME = os.environ.get("GLOBUS_APP_NAME", "inference_app")
AUTH_CLIENT_ID = os.environ.get("GLOBUS_AUTH_CLIENT_ID")
GATEWAY_CLIENT_ID = os.environ.get("GLOBUS_GATEWAY_CLIENT_ID")
GATEWAY_SCOPE = os.environ.get(
    "GLOBUS_GATEWAY_SCOPE",
    f"https://auth.globus.org/scopes/{GATEWAY_CLIENT_ID}/action_all",
)


def get_token() -> str:
    """Obtain a Globus access token for ALCF inference endpoints.

    Raises:
        RuntimeError: If no token can be obtained.
    """
    # 1. Explicit env-var
    token = os.environ.get("ALCF_ACCESS_TOKEN")
    if token:
        return token

    # 2. globus_sdk (preferred)
    try:
        import globus_sdk

        app = globus_sdk.UserApp(
            APP_NAME,
            client_id=AUTH_CLIENT_ID,
            scope_requirements={
                GATEWAY_CLIENT_ID: [GATEWAY_SCOPE],
            },
            config=globus_sdk.GlobusAppConfig(
                request_refresh_tokens=True,
            ),
        )
        auth = app.get_authorizer(GATEWAY_CLIENT_ID)
        auth.ensure_valid_token()
        logger.debug("[LLM] ALCF token obtained via globus_sdk")
        os.environ["ALCF_ACCESS_TOKEN"] = auth.access_token
        return auth.access_token
    except ImportError:
        logger.debug(
            "[LLM] globus_sdk not installed — install with: pip install aure[alcf]"
        )
    except Exception as exc:
        logger.warning("[LLM] globus_sdk token retrieval failed: %s", exc)

    # 3. subprocess fallback
    try:
        result = subprocess.run(
            ["python", "inference_auth_token.py", "get_access_token"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            os.environ["ALCF_ACCESS_TOKEN"] = result.stdout.strip()
            return result.stdout.strip()
        logger.warning(
            "[LLM] inference_auth_token.py exited with code %d: %s",
            result.returncode,
            result.stderr.strip(),
        )
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("[LLM] Failed to run inference_auth_token.py: %s", exc)

    raise RuntimeError(
        "Could not obtain ALCF access token. Set ALCF_ACCESS_TOKEN, "
        "install globus_sdk (pip install aure[alcf]) and authenticate, "
        "or download the auth script:\n"
        "  wget https://raw.githubusercontent.com/argonne-lcf/"
        "inference-endpoints/refs/heads/main/inference_auth_token.py\n"
        "  python inference_auth_token.py authenticate"
    )
