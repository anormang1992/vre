"""
Unit and integration tests for trace persistence.

Tests build_trace_entry, TraceWriter, and VRE integration.
"""

import json
from collections import deque
from uuid import UUID, uuid4

from vre import VRE
from vre.core.backends import Repository
from vre.core.grounding import GroundingResult
from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicQuery,
    EpistemicResponse,
    EpistemicResult,
    EpistemicStep,
    Primitive,
    PrimitiveMetrics,
    RelationType,
    ResolvedSubgraph,
)
import vre.tracing as tracing_module
from vre.tracing import TraceWriter, build_trace_entry


# ---------------------------------------------------------------------------
# Stub repository (matches test_vre.py pattern)
# ---------------------------------------------------------------------------

_TRANSITIVE_RELS = {RelationType.REQUIRES, RelationType.DEPENDS_ON, RelationType.CONSTRAINED_BY}


class StubRepository(Repository):
    def __init__(self, primitives: list[Primitive] | None = None) -> None:
        self._by_id: dict[UUID, Primitive] = {}
        self._by_name: dict[str, Primitive] = {}
        for p in primitives or []:
            self._by_id[p.id] = p
            self._by_name[p.name.lower()] = p

    def list_names(self) -> list[str]:
        return list(self._by_name.keys())

    def find_by_id(self, id: UUID) -> Primitive | None:
        return self._by_id.get(id)

    def find_by_name(self, name: str) -> Primitive | None:
        return self._by_name.get(name.lower())

    def save_primitive(self, primitive: Primitive) -> None:
        self._by_id[primitive.id] = primitive
        self._by_name[primitive.name.lower()] = primitive

    def update_metrics(self, primitive_id: UUID, metrics: PrimitiveMetrics) -> None:
        prim = self._by_id.get(primitive_id)
        if prim is not None:
            prim.metrics = metrics

    def batch_read_metrics(self, primitive_ids: list[UUID]) -> dict[UUID, PrimitiveMetrics | None]:
        return {pid: self._by_id[pid].metrics for pid in primitive_ids if pid in self._by_id}

    def batch_update_metrics(self, updates: dict[UUID, PrimitiveMetrics]) -> None:
        for pid, metrics in updates.items():
            prim = self._by_id.get(pid)
            if prim is not None:
                prim.metrics = metrics

    def resolve_subgraph(self, names: list[str]) -> ResolvedSubgraph:
        roots = [self._by_name[n.lower()] for n in names if n.lower() in self._by_name]
        visited: set[UUID] = {r.id for r in roots}
        queue: deque[UUID] = deque(r.id for r in roots)
        while queue:
            uid = queue.popleft()
            prim = self._by_id.get(uid)
            if not prim:
                continue
            for depth in prim.depths:
                for rel in depth.relata:
                    if rel.relation_type not in _TRANSITIVE_RELS:
                        continue
                    if rel.target_id not in visited:
                        visited.add(rel.target_id)
                        queue.append(rel.target_id)
        nodes = [self._by_id[uid] for uid in visited if uid in self._by_id]
        node_ids = {n.id for n in nodes}
        edges: list[EpistemicStep] = []
        for n in nodes:
            for depth in n.depths:
                for rel in depth.relata:
                    if rel.target_id not in node_ids:
                        continue
                    edges.append(EpistemicStep(
                        source_id=n.id,
                        target_id=rel.target_id,
                        relation_type=rel.relation_type,
                        source_depth=depth.level,
                        target_depth=rel.target_depth,
                    ))
        return ResolvedSubgraph(roots=roots, nodes=nodes, edges=edges)

    def delete_primitive(self, id: UUID) -> bool: raise NotImplementedError
    def clear(self) -> int: raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fully_grounded(name: str) -> Primitive:
    return Primitive(name=name, depths=[
        Depth(level=DepthLevel.EXISTENCE),
        Depth(level=DepthLevel.IDENTITY),
        Depth(level=DepthLevel.CAPABILITIES),
        Depth(level=DepthLevel.CONSTRAINTS),
    ])


def _grounding_result(
    grounded: bool = True,
    resolved: list[str] | None = None,
    gaps: list | None = None,
    with_trace: bool = True,
    agent_id: UUID | None = None,
) -> GroundingResult:
    """Build a GroundingResult with optional trace for testing."""
    resolved = resolved or ["file"]
    gaps = gaps or []

    trace = None
    if with_trace:
        file_id = uuid4()
        write_id = uuid4()
        step = EpistemicStep(
            source_id=write_id,
            target_id=file_id,
            relation_type=RelationType.APPLIES_TO,
            source_depth=DepthLevel.CAPABILITIES,
            target_depth=DepthLevel.CONSTRAINTS,
        )
        trace = EpistemicResponse(
            query=EpistemicQuery(concept_ids=[file_id, write_id]),
            result=EpistemicResult(
                primitives=[
                    Primitive(id=file_id, name="file"),
                    Primitive(id=write_id, name="write"),
                ],
                gaps=gaps,
                pathway=[step],
            ),
        )

    return GroundingResult(
        grounded=grounded,
        resolved=resolved,
        gaps=gaps,
        trace=trace,
        agent_id=agent_id,
    )


# ---------------------------------------------------------------------------
# build_trace_entry tests
# ---------------------------------------------------------------------------


class TestBuildTraceEntry:
    def test_check_operation(self):
        result = _grounding_result(grounded=True, resolved=["file", "write"])
        entry = build_trace_entry("check", ["file", "write"], result)

        assert entry.operation == "check"
        assert entry.concepts == ["file", "write"]
        assert entry.resolved == ["file", "write"]
        assert entry.grounded is True
        assert entry.gaps == []
        assert len(entry.steps) == 1
        assert entry.timestamp is not None

    def test_no_trace_gives_empty_steps(self):
        result = _grounding_result(with_trace=False)
        entry = build_trace_entry("check", ["file"], result)
        assert entry.steps == []

    def test_agent_id_serialized_as_string(self):
        aid = uuid4()
        result = _grounding_result(agent_id=aid)
        entry = build_trace_entry("check", ["file"], result)
        assert entry.agent_id == str(aid)

    def test_agent_id_none_when_absent(self):
        result = _grounding_result(agent_id=None)
        entry = build_trace_entry("check", ["file"], result)
        assert entry.agent_id is None

    def test_gaps_preserve_kind_discriminator(self):
        from vre.core.models import ExistenceGap

        gap_prim = Primitive(name="unknown")
        gap = ExistenceGap(primitive=gap_prim)
        result = _grounding_result(grounded=False, gaps=[gap])
        entry = build_trace_entry("check", ["unknown"], result)

        assert len(entry.gaps) == 1
        assert entry.gaps[0]["kind"] == "EXISTENCE"


# ---------------------------------------------------------------------------
# TraceEntry serialization tests
# ---------------------------------------------------------------------------


class TestTraceEntrySerialization:
    def test_roundtrip(self):
        result = _grounding_result()
        entry = build_trace_entry("check", ["file"], result)
        json_str = entry.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["operation"] == "check"
        assert parsed["concepts"] == ["file"]
        assert parsed["grounded"] is True
        assert isinstance(parsed["timestamp"], str)

    def test_each_field_present_in_json(self):
        result = _grounding_result()
        entry = build_trace_entry("check", ["file"], result)
        parsed = json.loads(entry.model_dump_json())

        expected_keys = {
            "timestamp", "operation", "concepts", "resolved",
            "grounded", "gaps", "steps", "agent_id",
        }
        assert set(parsed.keys()) == expected_keys


# ---------------------------------------------------------------------------
# TraceWriter tests
# ---------------------------------------------------------------------------


class TestTraceWriter:
    def test_write_creates_daily_file(self, tmp_path):
        writer = TraceWriter(tmp_path / "traces")
        entry = build_trace_entry("check", ["file"], _grounding_result())
        writer.write(entry)

        files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(files) == 1
        assert files[0].name.endswith(".jsonl")

    def test_write_appends_multiple_entries(self, tmp_path):
        writer = TraceWriter(tmp_path / "traces")
        entry1 = build_trace_entry("check", ["file"], _grounding_result())
        entry2 = build_trace_entry("check", ["write"], _grounding_result(resolved=["write"]))
        writer.write(entry1)
        writer.write(entry2)

        files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 2

    def test_creates_nested_directories(self, tmp_path):
        writer = TraceWriter(tmp_path / "deep" / "nested" / "traces")
        entry = build_trace_entry("check", ["file"], _grounding_result())
        writer.write(entry)

        assert (tmp_path / "deep" / "nested" / "traces").is_dir()
        files = list((tmp_path / "deep" / "nested" / "traces").glob("*.jsonl"))
        assert len(files) == 1

    def test_each_line_is_valid_json(self, tmp_path):
        writer = TraceWriter(tmp_path / "traces")
        for concept in ["file", "write", "read"]:
            entry = build_trace_entry("check", [concept], _grounding_result(resolved=[concept]))
            writer.write(entry)

        files = list((tmp_path / "traces").glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "operation" in parsed
            assert "concepts" in parsed


# ---------------------------------------------------------------------------
# VRE integration tests
# ---------------------------------------------------------------------------


class TestVRETraceIntegration:
    def test_check_writes_trace_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tracing_module, "DEFAULT_TRACE_DIR", tmp_path / "traces")
        file_p = _make_fully_grounded("file")
        repo = StubRepository([file_p])
        vre = VRE(repo)

        vre.check(["file"])

        files = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["operation"] == "check"
        assert parsed["concepts"] == ["file"]
        assert parsed["grounded"] is True

    def test_no_trace_when_disabled(self, tmp_path):
        file_p = _make_fully_grounded("file")
        repo = StubRepository([file_p])
        vre = VRE(repo, persist_traces=False)

        vre.check(["file"])

        assert not (tmp_path / "traces").exists()

    def test_trace_entry_content_structure(self, tmp_path, monkeypatch):
        """Verify all acceptance-criteria fields are present in the trace entry."""
        monkeypatch.setattr(tracing_module, "DEFAULT_TRACE_DIR", tmp_path / "traces")
        file_p = _make_fully_grounded("file")
        repo = StubRepository([file_p])
        vre = VRE(repo)

        vre.check(["file", "unknown_concept"])

        files = list((tmp_path / "traces").glob("*.jsonl"))
        parsed = json.loads(files[0].read_text().strip().split("\n")[0])

        assert "timestamp" in parsed
        assert parsed["concepts"] == ["file", "unknown_concept"]
        assert isinstance(parsed["resolved"], list)
        assert isinstance(parsed["grounded"], bool)
        assert isinstance(parsed["gaps"], list)
        assert isinstance(parsed["steps"], list)
        # Existence gap should have kind field
        existence_gaps = [g for g in parsed["gaps"] if g["kind"] == "EXISTENCE"]
        assert len(existence_gaps) >= 1
