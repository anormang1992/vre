# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Unit tests for vre.integrations.claude_code — install, uninstall, and _run_hook.
"""

import io
import json
import stat
from unittest.mock import MagicMock, patch

import pytest

from vre.core.grounding import GroundingResult
from vre.core.policy import PolicyResult


# ── helpers ──────────────────────────────────────────────────────────────────


def _stdin(payload: dict) -> io.StringIO:
    return io.StringIO(json.dumps(payload))


def _tool_payload(command: str = "") -> dict:
    return {"tool_input": {"command": command}}


# ── install ──────────────────────────────────────────────────────────────────


class TestInstall:

    def test_creates_config_with_restricted_permissions(self, tmp_path, monkeypatch):
        settings = tmp_path / ".claude" / "settings.json"
        config = tmp_path / ".vre" / "config.json"
        monkeypatch.setattr(
            "vre.integrations.claude_code._SETTINGS_PATH", settings
        )
        monkeypatch.setattr(
            "vre.integrations.claude_code._VRE_CONFIG_PATH", config
        )

        from vre.integrations.claude_code import install

        install("neo4j://localhost:7687", "neo4j", "pass")

        assert config.exists()
        mode = config.stat().st_mode & 0o777
        assert mode == 0o600

    def test_creates_claude_parent_directory(self, tmp_path, monkeypatch):
        settings = tmp_path / ".claude" / "settings.json"
        config = tmp_path / ".vre" / "config.json"
        monkeypatch.setattr(
            "vre.integrations.claude_code._SETTINGS_PATH", settings
        )
        monkeypatch.setattr(
            "vre.integrations.claude_code._VRE_CONFIG_PATH", config
        )

        from vre.integrations.claude_code import install

        install("neo4j://localhost:7687", "neo4j", "pass")

        assert settings.parent.is_dir()
        assert settings.exists()

    def test_idempotent_no_duplicate_hooks(self, tmp_path, monkeypatch):
        settings = tmp_path / ".claude" / "settings.json"
        config = tmp_path / ".vre" / "config.json"
        monkeypatch.setattr(
            "vre.integrations.claude_code._SETTINGS_PATH", settings
        )
        monkeypatch.setattr(
            "vre.integrations.claude_code._VRE_CONFIG_PATH", config
        )

        from vre.integrations.claude_code import install

        install("neo4j://localhost:7687", "neo4j", "pass")
        install("neo4j://localhost:7687", "neo4j", "pass")

        data = json.loads(settings.read_text())
        hooks = data["hooks"]["PreToolUse"]
        assert len(hooks) == 1


# ── uninstall ────────────────────────────────────────────────────────────────


class TestUninstall:

    def test_removes_hook_entry(self, tmp_path, monkeypatch):
        settings = tmp_path / ".claude" / "settings.json"
        config = tmp_path / ".vre" / "config.json"
        monkeypatch.setattr(
            "vre.integrations.claude_code._SETTINGS_PATH", settings
        )
        monkeypatch.setattr(
            "vre.integrations.claude_code._VRE_CONFIG_PATH", config
        )

        from vre.integrations.claude_code import install, uninstall

        install("neo4j://localhost:7687", "neo4j", "pass")
        uninstall()

        data = json.loads(settings.read_text())
        assert data["hooks"]["PreToolUse"] == []

    def test_safe_when_no_settings_file(self, tmp_path, monkeypatch):
        settings = tmp_path / ".claude" / "settings.json"
        monkeypatch.setattr(
            "vre.integrations.claude_code._SETTINGS_PATH", settings
        )

        from vre.integrations.claude_code import uninstall

        # Should not raise
        uninstall()


# ── _run_hook ────────────────────────────────────────────────────────────────


class TestRunHook:

    def test_allows_empty_command(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", _stdin(_tool_payload("")))

        from vre.integrations.claude_code import _run_hook

        with pytest.raises(SystemExit) as exc:
            _run_hook()

        assert exc.value.code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_allows_unrecognised_concepts(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.stdin", _stdin(_tool_payload("some_unknown_binary --flag"))
        )
        monkeypatch.setattr(
            "vre.builtins.shell.parse_bash_primitives", lambda _: []
        )

        from vre.integrations.claude_code import _run_hook

        with pytest.raises(SystemExit) as exc:
            _run_hook()

        assert exc.value.code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_blocks_ungrounded_concepts(self, tmp_path, monkeypatch, capsys):
        config = tmp_path / ".vre" / "config.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({
            "uri": "neo4j://localhost:7687",
            "user": "neo4j",
            "password": "pass",
            "database": "neo4j",
        }))
        monkeypatch.setattr(
            "vre.integrations.claude_code._VRE_CONFIG_PATH", config
        )
        monkeypatch.setattr(
            "sys.stdin", _stdin(_tool_payload("rm -rf /"))
        )
        monkeypatch.setattr(
            "vre.builtins.shell.parse_bash_primitives", lambda _: ["Delete"]
        )

        grounding = GroundingResult(grounded=False, resolved=["Delete"], gaps=[])
        mock_repo = MagicMock()
        mock_repo.__enter__ = MagicMock(return_value=mock_repo)
        mock_repo.__exit__ = MagicMock(return_value=False)

        mock_vre = MagicMock()
        mock_vre.check.return_value = grounding

        with patch("vre.core.graph.PrimitiveRepository", return_value=mock_repo), \
             patch("vre.VRE", return_value=mock_vre):
            from vre.integrations.claude_code import _run_hook

            with pytest.raises(SystemExit) as exc:
                _run_hook()

        assert exc.value.code == 2

    def test_defers_pending_policy_to_tui(self, tmp_path, monkeypatch, capsys):
        config = tmp_path / ".vre" / "config.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({
            "uri": "neo4j://localhost:7687",
            "user": "neo4j",
            "password": "pass",
            "database": "neo4j",
        }))
        monkeypatch.setattr(
            "vre.integrations.claude_code._VRE_CONFIG_PATH", config
        )
        monkeypatch.setattr(
            "sys.stdin", _stdin(_tool_payload("ls /etc"))
        )
        monkeypatch.setattr(
            "vre.builtins.shell.parse_bash_primitives", lambda _: ["Read"]
        )

        grounding = GroundingResult(grounded=True, resolved=["Read"], gaps=[])
        policy = PolicyResult(
            action="PENDING",
            confirmation_message="Read access requires confirmation.",
        )

        mock_repo = MagicMock()
        mock_repo.__enter__ = MagicMock(return_value=mock_repo)
        mock_repo.__exit__ = MagicMock(return_value=False)

        mock_vre = MagicMock()
        mock_vre.check.return_value = grounding
        mock_vre.check_policy.return_value = policy

        with patch("vre.core.graph.PrimitiveRepository", return_value=mock_repo), \
             patch("vre.VRE", return_value=mock_vre):
            from vre.integrations.claude_code import _run_hook

            with pytest.raises(SystemExit) as exc:
                _run_hook()

        assert exc.value.code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "ask"

    def test_fails_open_on_unexpected_exception(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("NOT JSON{{{"))

        from vre.integrations.claude_code import _run_hook

        with pytest.raises(SystemExit) as exc:
            _run_hook()

        assert exc.value.code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert "failing open" in out["hookSpecificOutput"].get("permissionDecisionReason", "")
