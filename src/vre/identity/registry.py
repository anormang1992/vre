# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
File-based agent registry.

Persists agent identities to a JSON file so that the same registration key
always resolves to the same stable UUID across process restarts.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from uuid import UUID

from vre.core.errors import RegistryError
from vre.identity.models import AgentIdentity


logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".vre" / "agents.json"


class AgentRegistry:
    """
    Append-only, file-based agent identity registry.

    Usage::

        registry = AgentRegistry()
        identity = registry.get_or_create("my-agent")
        # Same key → same UUID on every call, even across restarts.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else _DEFAULT_PATH

    # ── Private helpers ──────────────────────────────────────────────────

    def _load(self) -> dict[str, AgentIdentity]:
        """
        Load registry from disk. Returns empty dict if file doesn't exist.
        """
        if not self._path.exists():
            return {}

        try:
            raw = self._path.read_text(encoding="utf-8")
            if not raw.strip():
                return {}
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise RegistryError(
                    f"Corrupt registry at {self._path}: expected JSON object at top level"
                )
            registry = {
                key: AgentIdentity.model_validate(value) for key, value in data.items()
            }
            return registry
        except (json.JSONDecodeError, ValueError) as exc:
            raise RegistryError(f"Corrupt registry at {self._path}: {exc}") from exc
        except OSError as exc:
            raise RegistryError(f"Cannot read registry at {self._path}: {exc}") from exc

    def _save(self, registry: dict[str, AgentIdentity]) -> None:
        """
        Atomically write registry to disk. Creates parent dirs if needed.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        serialized = {
            key: identity.model_dump(mode="json") for key, identity in registry.items()
        }
        payload = json.dumps(serialized, indent=2)

        # Write to a temp file, fsync, then atomic rename. This ensures
        # agents.json is never left in a partial state: a crash during
        # write or fsync leaves the original file intact (the rename
        # hasn't happened yet), and a crash after fsync is safe because
        # the new data is already durable on disk before the swap.
        try:
            fd, tmp_path = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except OSError as exc:
            raise RegistryError(f"Cannot write registry to {self._path}: {exc}") from exc

    def get_or_create(self, registration_key: str, name: str | None = None) -> AgentIdentity:
        """
        Return the identity for *registration_key*, creating it if absent.

        Idempotent: repeated calls with the same key return the same identity
        (same `agent_id`). The *name* parameter is only used on first creation.
        """
        registry = self._load()

        if registration_key in registry:
            logger.debug("Registry hit for key '%s'", registration_key)
            agent_identity = registry[registration_key]
        else:
            agent_identity = AgentIdentity(registration_key=registration_key, name=name)
            registry[registration_key] = agent_identity
            self._save(registry)
            logger.info("Registered new agent '%s' → %s", registration_key, agent_identity.agent_id)

        return agent_identity

    def get_by_id(self, agent_id: UUID) -> AgentIdentity | None:
        """
        Look up an agent identity by its UUID.

        Returns None if no identity with the given `agent_id` exists.
        """
        registry = self._load()
        return next(
            (identity for identity in registry.values() if identity.agent_id == agent_id),
            None,
        )
