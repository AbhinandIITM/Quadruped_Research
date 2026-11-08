"""Unit tests for start_model_server authentication, bind security, and packaging.

Tests cover:
- Shared-secret auth check (disabled, valid, missing, wrong token)
- Constant-time comparison (hmac.compare_digest) usage
- Header extraction from Authorization: Bearer and X-Auth-Token
- Fail-closed refusal to bind a non-loopback host without a token
- Endpoint redaction so credentials never leak through /status
- pyproject [project.scripts] targets resolving to importable `main` callables
"""

import fnmatch
import hmac
import importlib
import os
import sys
import unittest
from unittest.mock import patch

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import start_model_server
from start_model_server import (
    InsecureBindError,
    check_request_authorized,
    extract_request_token,
    is_loopback_host,
    redact_endpoint_url,
    resolve_auth_token,
    validate_bind_security,
)

PYPROJECT_PATH = os.path.join(ROOT_DIR, "pyproject.toml")

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 has no stdlib tomllib
    tomllib = None


class TestAuthCheck(unittest.TestCase):
    """Auth resolution and request authorization semantics."""

    def test_auth_disabled_when_no_token_configured(self):
        authorized, _reason = check_request_authorized({}, None)
        self.assertTrue(authorized)

    def test_missing_token_is_rejected(self):
        authorized, reason = check_request_authorized({}, "s3cret")
        self.assertFalse(authorized)
        self.assertIn("missing", reason)

    def test_valid_bearer_token_is_accepted(self):
        authorized, reason = check_request_authorized(
            {"Authorization": "Bearer s3cret"}, "s3cret"
        )
        self.assertTrue(authorized)
        self.assertEqual(reason, "ok")

    def test_valid_x_auth_token_header_is_accepted(self):
        authorized, _reason = check_request_authorized({"X-Auth-Token": "s3cret"}, "s3cret")
        self.assertTrue(authorized)

    def test_wrong_token_is_rejected(self):
        for headers in ({"Authorization": "Bearer nope"}, {"X-Auth-Token": "nope"}):
            with self.subTest(headers=headers):
                authorized, reason = check_request_authorized(headers, "s3cret")
                self.assertFalse(authorized)
                self.assertEqual(reason, "invalid token")

    def test_malformed_authorization_header_is_rejected(self):
        # A non-Bearer scheme or an empty bearer value yields no extractable token.
        for headers in ({"Authorization": "Basic s3cret"}, {"Authorization": "Bearer "}):
            with self.subTest(headers=headers):
                authorized, _reason = check_request_authorized(headers, "s3cret")
                self.assertFalse(authorized)

    def test_uses_constant_time_comparison(self):
        with patch.object(
            start_model_server.hmac, "compare_digest", wraps=hmac.compare_digest
        ) as spy:
            check_request_authorized({"Authorization": "Bearer s3cret"}, "s3cret")
            spy.assert_called_once()

    def test_token_is_never_echoed_in_reason(self):
        _authorized, reason = check_request_authorized(
            {"Authorization": "Bearer hunter2"}, "s3cret"
        )
        self.assertNotIn("hunter2", reason)
        self.assertNotIn("s3cret", reason)

    def test_extract_request_token(self):
        self.assertEqual(extract_request_token({"Authorization": "Bearer abc"}), "abc")
        self.assertEqual(extract_request_token({"X-Auth-Token": "abc"}), "abc")
        self.assertIsNone(extract_request_token({}))
        self.assertIsNone(extract_request_token({"Authorization": "Basic abc"}))

    def test_resolve_auth_token_precedence(self):
        self.assertEqual(resolve_auth_token("cli", "env"), "cli")
        self.assertEqual(resolve_auth_token(None, "env"), "env")
        self.assertEqual(resolve_auth_token("", "env"), "env")
        self.assertIsNone(resolve_auth_token(None, None))
        self.assertIsNone(resolve_auth_token("  ", None))


class TestBindSecurity(unittest.TestCase):
    """Loopback detection and fail-closed non-loopback binding."""

    def test_loopback_hosts_are_detected(self):
        for host in ("127.0.0.1", "127.0.0.5", "::1", "localhost", "[::1]"):
            with self.subTest(host=host):
                self.assertTrue(is_loopback_host(host))

    def test_non_loopback_hosts_are_rejected(self):
        for host in ("0.0.0.0", "::", "192.168.1.10", "example.com", ""):
            with self.subTest(host=host):
                self.assertFalse(is_loopback_host(host))

    def test_non_loopback_without_token_refuses_to_start(self):
        for host in ("0.0.0.0", "::", "192.168.1.10"):
            with self.subTest(host=host):
                with self.assertRaises(InsecureBindError):
                    validate_bind_security(host, None)

    def test_loopback_without_token_is_allowed(self):
        validate_bind_security("127.0.0.1", None)

    def test_non_loopback_with_token_is_allowed(self):
        validate_bind_security("0.0.0.0", "s3cret")


class TestEndpointRedaction(unittest.TestCase):
    """Endpoint disclosure must not expose userinfo or query credentials."""

    def test_credentials_and_query_are_stripped(self):
        self.assertEqual(
            redact_endpoint_url("http://user:pass@gateway.local:8000/v1?api_key=leak"),
            "http://gateway.local:8000/v1",
        )

    def test_plain_endpoint_is_preserved(self):
        self.assertEqual(redact_endpoint_url("http://127.0.0.1:5005"), "http://127.0.0.1:5005")

    def test_missing_endpoint(self):
        self.assertEqual(redact_endpoint_url(None), "unknown")


class TestEntryPointWiring(unittest.TestCase):
    """Every declared console script must resolve to a real, importable main()."""

    def _load_pyproject(self) -> dict:
        if tomllib is None:  # pragma: no cover
            self.skipTest("tomllib requires Python 3.11+")
        with open(PYPROJECT_PATH, "rb") as handle:
            return tomllib.load(handle)

    def test_scripts_resolve_to_importable_main(self):
        scripts = self._load_pyproject().get("project", {}).get("scripts", {})
        self.assertTrue(scripts, "no [project.scripts] declared")
        for name, target in scripts.items():
            module_name, _, attr = target.partition(":")
            with self.subTest(script=name):
                self.assertTrue(module_name, f"{name} has no module target")
                self.assertEqual(attr, "main", f"{name} must target a main() callable")
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, attr, None)))

    def test_script_modules_are_packaged(self):
        data = self._load_pyproject()
        scripts = data.get("project", {}).get("scripts", {})
        py_modules = set(data.get("tool", {}).get("setuptools", {}).get("py-modules", []))
        include = (
            data.get("tool", {})
            .get("setuptools", {})
            .get("packages", {})
            .get("find", {})
            .get("include", [])
        )
        for name, target in scripts.items():
            module_name = target.split(":", 1)[0]
            top_level = module_name.split(".")[0]
            with self.subTest(script=name):
                packaged = module_name in py_modules or any(
                    fnmatch.fnmatchcase(top_level, pattern) for pattern in include
                )
                self.assertTrue(packaged, f"{module_name} is neither in py-modules nor included packages")


if __name__ == "__main__":
    unittest.main()
