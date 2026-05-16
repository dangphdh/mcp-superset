"""Authentication manager for Superset — JWT with CSRF and refresh."""

import re
import time

import httpx


class AuthManager:
    """Manages authentication with Superset REST API.

    Uses dual authentication flow:
    - Login: POST /api/v1/security/login with refresh=true
    - Session login: GET/POST /login/ (for read endpoints)
    - CSRF: GET /api/v1/security/csrf_token/ (required for POST/PUT/DELETE)
    - Refresh: POST /api/v1/security/refresh when access_token expires
    """

    _SESSION_VERIFY_Q = "(page:0,page_size:1)"

    def __init__(
        self,
        base_url: str,
        username: str | None = None,
        password: str | None = None,
        provider: str = "db",
    ):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.provider = provider

        # JWT state
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._csrf_token: str | None = None
        self._token_expires_at: float = 0
        self._session_authenticated = False

    async def get_token(self, client: httpx.AsyncClient) -> str:
        """Return a valid access_token, refreshing or re-logging in as needed.

        Args:
            client: httpx async client used for HTTP requests.

        Returns:
            A valid JWT access token string.
        """
        # Check if token is still valid (with 30 sec safety margin)
        if self._access_token and time.time() < self._token_expires_at - 30:
            return self._access_token

        # Try refresh if we have a refresh token
        if self._refresh_token:
            refreshed = await self._refresh(client)
            if refreshed:
                return self._access_token

        # Full login
        await self._login(client)
        return self._access_token

    async def get_csrf_token(self, client: httpx.AsyncClient) -> str:
        """Return a valid CSRF token, fetching one if necessary.

        Args:
            client: httpx async client used for HTTP requests.

        Returns:
            A CSRF token string.
        """
        if self._csrf_token:
            return self._csrf_token
        await self._fetch_csrf(client)
        return self._csrf_token

    async def ensure_session(self, client: httpx.AsyncClient) -> None:
        """Ensure Flask session auth is established for read requests."""
        if self._session_authenticated:
            return
        await self._login_form(client)

    async def _login(self, client: httpx.AsyncClient) -> None:
        """Perform JWT login via POST /api/v1/security/login.

        Args:
            client: httpx async client used for HTTP requests.
        """
        url = f"{self.base_url}/api/v1/security/login"
        payload = {
            "username": self.username,
            "password": self.password,
            "provider": self.provider,
            "refresh": True,
        }
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._refresh_token = data.get("refresh_token")
        # Default JWT_ACCESS_TOKEN_EXPIRES = 15 minutes (900 sec)
        self._token_expires_at = time.time() + 900
        # Reset CSRF — it is bound to the session/token
        self._csrf_token = None

    async def _login_form(self, client: httpx.AsyncClient) -> None:
        """Perform form-based login via /login/ to obtain session cookie."""
        login_url = f"{self.base_url}/login/"

        login_page = await client.get(login_url)
        login_page.raise_for_status()

        # Superset login template renders csrf_token as a hidden input.
        # We extract it directly to avoid adding an HTML parser dependency.
        # Supports both attribute orders: name->value and value->name.
        match = re.search(
            r"(?:name=['\"]csrf_token['\"][^>]*value=['\"]([^'\"]+)['\"]|value=['\"]([^'\"]+)['\"][^>]*name=['\"]csrf_token['\"])",
            login_page.text,
        )
        if not match:
            raise ValueError("Unable to extract csrf_token from Superset login page")
        csrf_token = match.group(1) or match.group(2)

        resp = await client.post(
            login_url,
            data={
                "csrf_token": csrf_token,
                "username": self.username,
                "password": self.password,
            },
        )
        resp.raise_for_status()

        # RISON q is required by Superset list endpoints; this lightweight call
        # confirms session-cookie auth is active for read requests.
        verify_session = await client.get(
            f"{self.base_url}/api/v1/dashboard/",
            params={"q": self._SESSION_VERIFY_Q},
        )
        if verify_session.status_code in (401, 302):
            raise httpx.HTTPStatusError(
                f"Superset form login failed to establish session (status={verify_session.status_code})",
                request=verify_session.request,
                response=verify_session,
            )

        self._session_authenticated = True

    async def _refresh(self, client: httpx.AsyncClient) -> bool:
        """Attempt to refresh the JWT using the refresh token.

        Args:
            client: httpx async client used for HTTP requests.

        Returns:
            True if refresh succeeded, False otherwise.
        """
        url = f"{self.base_url}/api/v1/security/refresh"
        headers = {"Authorization": f"Bearer {self._refresh_token}"}
        try:
            resp = await client.post(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            self._access_token = data["access_token"]
            self._token_expires_at = time.time() + 900
            # Reset CSRF — a new one is needed for the new token
            self._csrf_token = None
            return True
        except (httpx.HTTPStatusError, KeyError):
            # Refresh failed — full login required
            self._refresh_token = None
            return False

    async def _fetch_csrf(self, client: httpx.AsyncClient) -> None:
        """Fetch CSRF token via GET /api/v1/security/csrf_token/.

        Args:
            client: httpx async client used for HTTP requests.
        """
        token = await self.get_token(client)
        url = f"{self.base_url}/api/v1/security/csrf_token/"
        headers = {"Authorization": f"Bearer {token}"}
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        self._csrf_token = data["result"]

    def invalidate(self) -> None:
        """Reset all cached tokens, forcing re-authentication on next request."""
        self._access_token = None
        self._refresh_token = None
        self._csrf_token = None
        self._token_expires_at = 0
        self._session_authenticated = False
