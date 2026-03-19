# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Unit tests for vre.integrations.claude_code — install, uninstall, and _run_hook.
"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from vre.core.grounding import GroundingResult
from vre.core.policy import PolicyAction, PolicyResult
from vre.core.policy.models import Policy, PolicyViolation


# ── helpers ──────────────────────────────────────────────────────────────────


def _stdin(payload: dict) -> io.StringIO:
    return io.StringIO(json.dumps(payload))


def _tool_payload(command: str = "") -> dict:
    return {"tool_input": {"command": command}}


def _prefixed_command(concepts: str, command: str) -> str:
    return f"# vre:{concepts}\n{command}"


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


# ── _parse_vre_prefix ───────────────────────────────────────────────────────


class TestParseVrePrefix:

    def test_extracts_concepts_and_clean_command(self):
        from vre.integrations.claude_code import _parse_vre_prefix

        result = _parse_vre_prefix("# vre:delete,file\nrm -rf foo/")
        assert result is not None
        concepts, clean = result
        assert set(concepts) == {"delete", "file"}
        assert clean == "rm -rf foo/"

    def test_returns_none_without_prefix(self):
        from vre.integrations.claude_code import _parse_vre_prefix

        assert _parse_vre_prefix("rm -rf foo/") is None

    def test_normalizes_to_lowercase(self):
        from vre.integrations.claude_code import _parse_vre_prefix

        result = _parse_vre_prefix("# vre:Delete,File\nrm foo")
        assert result is not None
        concepts, _ = result
        assert concepts == ["delete", "file"]

    def test_handles_spaces_around_concepts(self):
        from vre.integrations.claude_code import _parse_vre_prefix

        result = _parse_vre_prefix("# vre: delete , file \nrm foo")
        assert result is not None
        concepts, _ = result
        assert set(concepts) == {"delete", "file"}

    def test_handles_no_newline(self):
        from vre.integrations.claude_code import _parse_vre_prefix

        result = _parse_vre_prefix("# vre:read")
        assert result is not None
        concepts, clean = result
        assert concepts == ["read"]
        assert clean == ""


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

    def test_blocks_command_without_prefix(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.stdin", _stdin(_tool_payload("rm -rf /"))
        )

        from vre.integrations.claude_code import _run_hook

        with pytest.raises(SystemExit) as exc:
            _run_hook()

        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "# vre:" in err
        assert "conceptual primitives" in err

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
            "sys.stdin",
            _stdin(_tool_payload(_prefixed_command("delete,file", "rm -rf /"))),
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

    def test_allows_grounded_with_updated_input(self, tmp_path, monkeypatch, capsys):
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
            "sys.stdin",
            _stdin(_tool_payload(_prefixed_command("read,file", "cat foo.txt"))),
        )

        grounding = GroundingResult(grounded=True, resolved=["Read", "File"], gaps=[])
        policy = PolicyResult(action=PolicyAction.PASS, reason="No violations", violations=[])

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
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert out["hookSpecificOutput"]["updatedInput"]["command"] == "cat foo.txt"

    def test_defers_confirmation_required_policy_to_tui(self, tmp_path, monkeypatch, capsys):
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
            "sys.stdin",
            _stdin(_tool_payload(_prefixed_command("read", "ls /etc"))),
        )

        grounding = GroundingResult(grounded=True, resolved=["Read"], gaps=[])
        violation = PolicyViolation(
            policy=Policy(name="ReadPolicy", requires_confirmation=True),
            message="Read access requires confirmation.",
        )
        policy = PolicyResult(
            action=PolicyAction.BLOCK,
            reason="Confirmation required, no handler",
            violations=[violation],
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
        assert "updatedInput" not in out["hookSpecificOutput"]

    def test_fails_open_on_unexpected_exception(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("NOT JSON{{{"))

        from vre.integrations.claude_code import _run_hook

        with pytest.raises(SystemExit) as exc:
            _run_hook()

        assert exc.value.code == 0
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert "failing open" in out["hookSpecificOutput"].get("permissionDecisionReason", "")
