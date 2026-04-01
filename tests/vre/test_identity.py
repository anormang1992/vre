"""
Unit tests for vre.identity — AgentIdentity model and AgentRegistry.
"""

import json
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from vre.core.errors import RegistryError
from vre.identity.models import AgentIdentity
from vre.identity.registry import AgentRegistry


# ── AgentIdentity model ─────────────────────────────────────────────────────


class TestAgentIdentity:

    def test_auto_generates_uuid(self):
        identity = AgentIdentity(registration_key="test-agent")
        assert isinstance(identity.agent_id, UUID)

    def test_registration_key_required(self):
        with pytest.raises(ValidationError):
            AgentIdentity()

    def test_name_defaults_to_none(self):
        identity = AgentIdentity(registration_key="test-agent")
        assert identity.name is None

    def test_name_can_be_set(self):
        identity = AgentIdentity(registration_key="test-agent", name="My Agent")
        assert identity.name == "My Agent"

    def test_created_at_auto_set(self):
        identity = AgentIdentity(registration_key="test-agent")
        assert identity.created_at is not None

    def test_serialization_roundtrip(self):
        identity = AgentIdentity(registration_key="test-agent", name="Agent A")
        dumped = identity.model_dump(mode="json")
        restored = AgentIdentity.model_validate(dumped)
        assert restored.agent_id == identity.agent_id
        assert restored.registration_key == identity.registration_key
        assert restored.name == identity.name

    def test_two_identities_have_different_uuids(self):
        a = AgentIdentity(registration_key="a")
        b = AgentIdentity(registration_key="b")
        assert a.agent_id != b.agent_id


# ── AgentRegistry ────────────────────────────────────────────────────────────


class TestAgentRegistry:

    def test_creates_new_identity(self, tmp_path):
        registry = AgentRegistry(path=tmp_path / "agents.json")
        identity = registry.get_or_create("my-agent")
        assert isinstance(identity.agent_id, UUID)
        assert identity.registration_key == "my-agent"

    def test_idempotent_same_uuid(self, tmp_path):
        registry = AgentRegistry(path=tmp_path / "agents.json")
        first = registry.get_or_create("my-agent")
        second = registry.get_or_create("my-agent")
        assert first.agent_id == second.agent_id

    def test_different_keys_different_uuids(self, tmp_path):
        registry = AgentRegistry(path=tmp_path / "agents.json")
        a = registry.get_or_create("agent-a")
        b = registry.get_or_create("agent-b")
        assert a.agent_id != b.agent_id

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "agents.json"
        first_id = AgentRegistry(path=path).get_or_create("my-agent").agent_id
        second_id = AgentRegistry(path=path).get_or_create("my-agent").agent_id
        assert first_id == second_id

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "agents.json"
        registry = AgentRegistry(path=path)
        registry.get_or_create("my-agent")
        assert path.exists()

    def test_name_stored_on_creation(self, tmp_path):
        registry = AgentRegistry(path=tmp_path / "agents.json")
        identity = registry.get_or_create("my-agent", name="My Agent")
        assert identity.name == "My Agent"

    def test_name_not_overwritten_on_subsequent_calls(self, tmp_path):
        registry = AgentRegistry(path=tmp_path / "agents.json")
        registry.get_or_create("my-agent", name="First Name")
        second = registry.get_or_create("my-agent", name="Second Name")
        assert second.name == "First Name"

    def test_corrupt_json_raises_registry_error(self, tmp_path):
        path = tmp_path / "agents.json"
        path.write_text("not valid json", encoding="utf-8")
        registry = AgentRegistry(path=path)
        with pytest.raises(RegistryError, match="Corrupt registry"):
            registry.get_or_create("my-agent")

    def test_non_dict_json_raises_registry_error(self, tmp_path):
        path = tmp_path / "agents.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        registry = AgentRegistry(path=path)
        with pytest.raises(RegistryError, match="expected JSON object"):
            registry.get_or_create("my-agent")

    def test_empty_file_treated_as_empty_registry(self, tmp_path):
        path = tmp_path / "agents.json"
        path.write_text("", encoding="utf-8")
        registry = AgentRegistry(path=path)
        identity = registry.get_or_create("my-agent")
        assert isinstance(identity.agent_id, UUID)

    def test_get_by_id_returns_matching_identity(self, tmp_path):
        registry = AgentRegistry(path=tmp_path / "agents.json")
        created = registry.get_or_create("my-agent")
        found = registry.get_by_id(created.agent_id)
        assert found is not None
        assert found.agent_id == created.agent_id
        assert found.registration_key == "my-agent"

    def test_get_by_id_returns_none_for_unknown_uuid(self, tmp_path):
        registry = AgentRegistry(path=tmp_path / "agents.json")
        registry.get_or_create("my-agent")
        assert registry.get_by_id(uuid4()) is None

    def test_get_by_id_returns_none_on_empty_registry(self, tmp_path):
        registry = AgentRegistry(path=tmp_path / "agents.json")
        assert registry.get_by_id(uuid4()) is None

    def test_registry_file_is_valid_json(self, tmp_path):
        path = tmp_path / "agents.json"
        registry = AgentRegistry(path=path)
        registry.get_or_create("agent-a")
        registry.get_or_create("agent-b")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "agent-a" in data
        assert "agent-b" in data
