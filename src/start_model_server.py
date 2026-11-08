"""Dedicated Model Server for Go2 CHAMP Natural Language Control.

This standalone service defines, initializes, and serves a specific AI model
(via OmniRoute, Ollama, Google Gemini, or Mock) as a lightweight local HTTP
microservice (default: http://127.0.0.1:5005).

The model configuration is strictly encapsulated in this script, allowing
`go2_champ_llm_cli.py` to communicate directly with this live running service
without needing to know or specify the underlying model.

Usage:
    # 1. Start with configured defaults (OmniRoute / OpenAI-compatible):
    python start_model_server.py

    # 2. Start with a specific model on OmniRoute:
    python start_model_server.py --model my-custom-model --url http://localhost:8000/v1

    # 3. Start with local Ollama:
    python start_model_server.py --backend ollama --model deepseek-r1:8b

    # 4. Start with Google Gemini:
    python start_model_server.py --backend gemini --model gemini-2.5-flash

    # 5. Require a shared secret on POST /query (mandatory for a non-loopback bind):
    python start_model_server.py --host 0.0.0.0 --token "$MODEL_SERVER_TOKEN"
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
import socketserver
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

# Ensure root directory is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from go2_llm_commands import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OMNIROUTE_API_KEY,
    DEFAULT_OMNIROUTE_MODEL,
    DEFAULT_OMNIROUTE_URL,
    GeminiClient,
    LLMCommandResult,
    MockDeepSeekClient,
    OllamaDeepSeekClient,
    OmniRouteClient,
    load_dotenv,
)

# Load .env file automatically
load_dotenv()

# =============================================================================
# MODEL DEFINITION & CONFIGURATION (Defined strictly in this script)
# =============================================================================

ACTIVE_BACKEND: str = os.environ.get("MODEL_BACKEND", "omniroute")
ACTIVE_MODEL: str = os.environ.get("OMNIROUTE_MODEL", DEFAULT_OMNIROUTE_MODEL)
OMNIROUTE_URL: str = os.environ.get("OMNIROUTE_BASE_URL", DEFAULT_OMNIROUTE_URL)
OMNIROUTE_API_KEY: str = os.environ.get("OMNIROUTE_API_KEY", DEFAULT_OMNIROUTE_API_KEY)

# Server Host & Port
SERVER_HOST: str = "127.0.0.1"
SERVER_PORT: int = int(os.environ.get("MODEL_SERVER_PORT", "5005"))

# Shared-secret authentication. When unset, POST /query is open, but only on a
# loopback bind (a non-loopback bind without a token is refused at startup).
TOKEN_ENV_VAR: str = "MODEL_SERVER_TOKEN"
SERVER_TOKEN: Optional[str] = os.environ.get(TOKEN_ENV_VAR) or None


# -----------------------------------------------------------------------------
# Authentication & Bind-Security Helpers
# -----------------------------------------------------------------------------

class InsecureBindError(ValueError):
    """Raised when the server would bind to a network-reachable interface with no token."""


def is_loopback_host(host: str) -> bool:
    """Return True when ``host`` names a loopback interface (127.0.0.0/8, ::1, localhost)."""
    if not host:
        return False
    candidate = host.strip().strip("[]")
    if candidate.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def resolve_auth_token(cli_token: Optional[str], env_token: Optional[str] = None) -> Optional[str]:
    """Resolve the shared secret, with an explicit CLI token taking precedence over the env var."""
    token = cli_token if cli_token else env_token
    token = (token or "").strip()
    return token or None


def extract_request_token(headers: Any) -> Optional[str]:
    """Extract a shared secret from ``Authorization: Bearer <token>`` or ``X-Auth-Token``."""
    auth = (headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
        if candidate:
            return candidate
    return (headers.get("X-Auth-Token") or "").strip() or None


def check_request_authorized(headers: Any, configured_token: Optional[str]) -> Tuple[bool, str]:
    """Validate an incoming request against the configured secret.

    Returns ``(authorized, reason)``. Never echoes the secret material.
    """
    if not configured_token:
        return True, "auth-disabled"
    provided = extract_request_token(headers)
    if not provided:
        return False, "missing token"
    if not hmac.compare_digest(provided.encode("utf-8"), configured_token.encode("utf-8")):
        return False, "invalid token"
    return True, "ok"


def validate_bind_security(host: str, token: Optional[str]) -> None:
    """Fail closed: a non-loopback bind requires a token.

    Raises:
        InsecureBindError: If ``host`` is network-reachable and no token is configured.
    """
    if token:
        return
    if not is_loopback_host(host):
        raise InsecureBindError(
            f"Refusing to bind to non-loopback host '{host}' without authentication. "
            f"Set a shared secret via --token or the {TOKEN_ENV_VAR} environment variable, "
            "or bind to 127.0.0.1 for local-only access."
        )


def redact_endpoint_url(endpoint: Optional[str]) -> str:
    """Strip credentials, query strings and fragments before an endpoint is disclosed."""
    if not endpoint:
        return "unknown"
    try:
        parts = urlsplit(endpoint)
    except ValueError:
        return "<redacted>"
    if not parts.scheme or not parts.netloc:
        return endpoint
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, "", ""))


# -----------------------------------------------------------------------------
# Threaded HTTP Server & Request Handler
# -----------------------------------------------------------------------------

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ModelServiceHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for model query and health-check endpoints."""

    client: Any = None
    model_name: str = "unknown"
    backend_name: str = "unknown"
    endpoint_url: str = "unknown"
    auth_token: Optional[str] = None

    def log_message(self, _format: str, *_args: Any) -> None:
        """Suppress default HTTP server access logs for cleaner console output."""
        return

    def _send_json(
        self,
        status_code: int,
        payload: dict,
        extra_headers: Optional[dict] = None,
    ) -> None:
        """Write a JSON response with an explicit status code."""
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for header, value in (extra_headers or {}).items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_unauthorized(self) -> None:
        """Reject a request that failed authentication (never echoes the secret)."""
        self._send_json(
            401,
            {
                "status": "error",
                "error": "unauthorized",
                "message": "Missing or invalid authentication token.",
            },
            extra_headers={"WWW-Authenticate": "Bearer"},
        )

    def do_GET(self) -> None:
        """Handle GET requests (/health, /status, /).

        ``/health`` is an unauthenticated liveness probe and discloses no secrets.
        ``/status`` requires the token when one is configured and only ever reports a
        redacted endpoint.
        """
        if self.path == "/health":
            uptime = round(time.perf_counter() - getattr(self.server, "start_time", 0.0), 1)
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model": self.model_name,
                    "backend": self.backend_name,
                    "auth_required": bool(self.auth_token),
                    "uptime_s": uptime,
                    "server_time": datetime.now(timezone.utc).isoformat(),
                },
            )
        elif self.path == "/status":
            authorized, _reason = check_request_authorized(self.headers, self.auth_token)
            if not authorized:
                self._send_unauthorized()
                return
            uptime = round(time.perf_counter() - getattr(self.server, "start_time", 0.0), 1)
            self._send_json(
                200,
                {
                    "status": "ok",
                    "backend": self.backend_name,
                    "model": self.model_name,
                    "endpoint": redact_endpoint_url(self.endpoint_url),
                    "auth_required": bool(self.auth_token),
                    "uptime_s": uptime,
                    "server_time": datetime.now(timezone.utc).isoformat(),
                },
            )
        else:
            msg = f"Go2 CHAMP Model Server is running.\nActive Model: {self.model_name} ({self.backend_name})\n"
            body = msg.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self) -> None:
        """Handle POST requests (/query)."""
        if self.path != "/query":
            self.send_error(404, "Endpoint not found. Use POST /query")
            return

        authorized, reason = check_request_authorized(self.headers, self.auth_token)
        if not authorized:
            timestamp_str = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp_str}] [Query Rejected] Unauthorized request ({reason}).")
            self._send_unauthorized()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self.send_error(400, "Empty request body")
            return

        raw_body = self.rfile.read(content_length)
        try:
            req_data = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            self.send_error(400, f"Invalid JSON payload: {e}")
            return

        prompt = req_data.get("prompt", "").strip()
        system_prompt = req_data.get("system_prompt", None)

        if not prompt:
            self.send_error(400, "Missing 'prompt' in request payload")
            return

        timestamp_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp_str}] [Query Received] \"{prompt}\"")

        t0 = time.perf_counter()
        try:
            result: LLMCommandResult = self.client.query(
                user_prompt=prompt,
                system_prompt=system_prompt,
            )
            elapsed = time.perf_counter() - t0
            cmd_count = len(result.commands)
            print(f"[{timestamp_str}] [Query Completed in {elapsed:.2f}s] Status: {result.status.upper()} | Commands: {cmd_count}")
            for idx, cmd in enumerate(result.commands):
                if cmd.mode.name == "WAYPOINT" and cmd.waypoint:
                    print(f"         └─ [{idx}] {cmd.mode.name:<8} target=({cmd.waypoint.x:+.2f}, {cmd.waypoint.y:+.2f})m speed={cmd.waypoint.speed:.2f}m/s dur={cmd.duration:.1f}s")
                else:
                    print(f"         └─ [{idx}] {cmd.mode.name:<8} vx={cmd.vx:+.2f} vy={cmd.vy:+.2f} yaw={cmd.yaw_rate:+.2f} dur={cmd.duration:.1f}s")

            resp_dict = result.to_audit_dict()
            resp_body = json.dumps(resp_dict).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except Exception as err:
            elapsed = time.perf_counter() - t0
            print(f"[{timestamp_str}] [Query Error in {elapsed:.2f}s]: {err}")
            err_payload = {
                "status": "error",
                "explanation": f"Server query execution error: {err}",
                "commands": [{"mode": "stand", "duration": 1.0, "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}],
                "error_message": str(err),
                "model_name": self.model_name,
                "wall_time_s": elapsed,
            }
            resp_body = json.dumps(err_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)


# -----------------------------------------------------------------------------
# Main Server Runner
# -----------------------------------------------------------------------------

def start_server(
    backend: str = ACTIVE_BACKEND,
    model: Optional[str] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    host: str = SERVER_HOST,
    port: int = SERVER_PORT,
    token: Optional[str] = None,
) -> None:
    """Initialize the configured model client and start the HTTP server.

    A shared secret (``--token`` or ``MODEL_SERVER_TOKEN``) is required for POST
    /query. Binding to a non-loopback host with no token configured is refused.
    """
    token = resolve_auth_token(token, SERVER_TOKEN)
    try:
        validate_bind_security(host, token)
    except InsecureBindError as err:
        print(f"\n[Error] {err}\n")
        sys.exit(1)

    backend = backend.lower().strip()
    client: Any = None
    model_name: str = ""
    endpoint_url: str = ""

    if backend == "omniroute":
        model_name = model or ACTIVE_MODEL
        endpoint_url = url or OMNIROUTE_URL
        key = api_key if api_key is not None else OMNIROUTE_API_KEY
        client = OmniRouteClient(base_url=endpoint_url, model=model_name, api_key=key)
        backend_display = "OmniRoute (OpenAI-Compatible Gateway)"
    elif backend == "ollama":
        model_name = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL_NAME)
        endpoint_url = url or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        client = OllamaDeepSeekClient(host=endpoint_url, model=model_name)
        backend_display = "Local Ollama"
    elif backend == "gemini":
        model_name = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        endpoint_url = "https://generativelanguage.googleapis.com"
        client = GeminiClient(api_key=key, model=model_name)
        backend_display = "Google AI Studio / Gemini API"
    elif backend == "smolvla":
        from go2_smolvla import SmolVLAQuadrupedAgent
        model_name = model or "lerobot/smolvla_base"
        endpoint_url = "In-Process PyTorch GPU/CPU"
        client = SmolVLAQuadrupedAgent(model_id=model_name, load_pretrained=False)
        backend_display = "Hugging Face SmolVLA (Local VLA Policy)"
    elif backend == "mock":
        model_name = "mock-client"
        endpoint_url = "local-offline"
        client = MockDeepSeekClient()
        backend_display = "Offline Mock Client"
    else:
        print(f"Error: Unknown backend '{backend}'. Choose from: omniroute, ollama, gemini, smolvla, mock.")
        sys.exit(1)

    # Check connection
    available, msg = client.check_availability()
    if not available:
        print(f"[Warning] {msg}")
        print("Starting server anyway (queries will fail closed to safe stand if endpoint remains offline).")
    else:
        print(f"✓ {msg}")

    # Bind handler class attributes
    ModelServiceHandler.client = client
    ModelServiceHandler.model_name = model_name
    ModelServiceHandler.backend_name = backend_display
    ModelServiceHandler.endpoint_url = endpoint_url
    ModelServiceHandler.auth_token = token

    server_address = (host, port)
    try:
        httpd = ThreadedHTTPServer(server_address, ModelServiceHandler)
    except OSError as err:
        print(f"\n[Error] Failed to bind server on {host}:{port}: {err}")
        print("Tip: If the port is already in use, specify another with --port <number>.\n")
        sys.exit(1)

    httpd.start_time = time.perf_counter()

    print("\n" + "=" * 65)
    print(" 🚀 Go2 CHAMP Model Server is Running!")
    print("=" * 65)
    print(f" Active Model:  {model_name}  (Defined in start_model_server.py)")
    print(f" Backend:       {backend_display}")
    print(f" Endpoint URL:  {endpoint_url}")
    print(f" Service URL:   http://{host}:{port}")
    if token:
        print(" Auth:          Bearer token required for POST /query")
    else:
        print(" Auth:          disabled (loopback-only bind; set --token to require one)")
    print("-----------------------------------------------------------------")
    print(" Ready to receive natural language prompts from:")
    print(f"   python go2_champ_llm_cli.py --prompt \"Walk forward 2 meters\"")
    print(f"   python go2_champ_llm_cli.py --interactive")
    print("-----------------------------------------------------------------")
    print(" Press Ctrl+C to stop the model server.")
    print("=" * 65 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Shutting down] Stopping Go2 CHAMP Model Server...")
        httpd.shutdown()
        httpd.server_close()
        print("[Shutdown complete] Goodbye.")


def main():
    parser = argparse.ArgumentParser(description="Start dedicated Model Server for Go2 CHAMP Natural Language Control")
    parser.add_argument("--backend", choices=["omniroute", "ollama", "gemini", "smolvla", "mock"], default=ACTIVE_BACKEND, help=f"Backend provider (default: {ACTIVE_BACKEND})")
    parser.add_argument("--model", type=str, default=None, help=f"Model name to serve (default from script: {ACTIVE_MODEL})")
    parser.add_argument("--url", type=str, default=None, help="Base endpoint URL (for OmniRoute or Ollama)")
    parser.add_argument("--api-key", type=str, default=None, help="API key (if required by OmniRoute or Gemini)")
    parser.add_argument("--host", type=str, default=SERVER_HOST, help=f"Server host IP (default: {SERVER_HOST})")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help=f"Server port (default: {SERVER_PORT})")
    parser.add_argument("--token", type=str, default=None, help=f"Shared secret required for POST /query (env: {TOKEN_ENV_VAR})")
    args = parser.parse_args()

    start_server(
        backend=args.backend,
        model=args.model,
        url=args.url,
        api_key=args.api_key,
        host=args.host,
        port=args.port,
        token=args.token,
    )


if __name__ == "__main__":
    main()
