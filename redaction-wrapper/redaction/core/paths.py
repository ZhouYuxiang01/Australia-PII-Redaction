"""Environment-variable substitution for config files.

Supports ``${VAR}`` and ``${VAR:-default}`` placeholders inside any string in a
loaded JSON config. This lets backend configs ship without hardcoded
``/home/admin/...`` paths — deployers point env vars at their model checkouts.

The substitution is recursive over dicts/lists/strings. Non-string leaves are
returned unchanged. A ``${VAR}`` with no matching env var and no default is
left as-is (so the existing missing-file error surfaces with the placeholder
visible — easier to diagnose than a silent empty path).
"""
from __future__ import annotations

import os
import re
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand_string(value: str, env: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        if name in env and env[name] != "":
            return env[name]
        if default is not None:
            return default
        return m.group(0)
    return _PLACEHOLDER_RE.sub(repl, value)


def expand_env_placeholders(obj: Any, env: dict[str, str] | None = None) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in any string leaves."""
    e = env if env is not None else dict(os.environ)
    if isinstance(obj, str):
        return _expand_string(obj, e)
    if isinstance(obj, dict):
        return {k: expand_env_placeholders(v, e) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_placeholders(v, e) for v in obj]
    return obj
