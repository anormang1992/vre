"""
Unit tests for GroundingResult.__str__ and PolicyResult.__str__.
"""

from uuid import uuid4

from vre.core.models import (
    Depth,
    DepthLevel,
    EpistemicQuery,
    EpistemicResult,
    EpistemicResponse,
    EpistemicStep,
    ExistenceGap,
    DepthGap,
    RelationalGap,
    ReachabilityGap,
    Primitive,
    Relatum,
    RelationType,
)
from vre.core.grounding import GroundingResult
from vre.core.policy import PolicyResult


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_primitive(name: str, depths: list | None = None) -> Primitive:
    return Primitive(name=name, depths=depths or [])


def _make_depth(
    level: DepthLevel,
    properties: dict | None = None,
    relata: list | None = None,
) -> Depth:
    return Depth(level=level, properties=properties or {}, relata=relata or [])


def _make_trace(
    primitives: list,
    pathway: list | None = None,
) -> EpistemicResponse:
    return EpistemicResponse(
        query=EpistemicQuery(concept_ids=[p.id for p in primitives]),
        result=EpistemicResult(primitives=primitives, pathway=pathway or []),
    )


def _grounded(resolved: list[str], primitives: list, pathway: list | None = None) -> GroundingResult:
    return GroundingResult(
        grounded=True,
        resolved=resolved,
        gaps=[],
        trace=_make_trace(primitives, pathway),
    )


# ── GroundingResult.__str__ ───────────────────────────────────────────────────

def test_grounding_str_grounded_header():
    p = _make_primitive("file")
    result = _grounded(["file"], [p])
    s = str(result)
    assert s.startswith("[VRE] Grounded — file")
    assert "This is your epistemic trace" in s


def test_grounding_str_shows_depth_labels_and_properties():
    p = _make_primitive("file", [
        _make_depth(DepthLevel.EXISTENCE),
        _make_depth(DepthLevel.IDENTITY, properties={"description": "A file"}),
        _make_depth(DepthLevel.CAPABILITIES),
        _make_depth(DepthLevel.CONSTRAINTS),
    ])
    result = _grounded(["file"], [p])
    s = str(result)
    assert "D1 IDENTITY" in s
    assert "description: A file" in s


def test_grounding_str_shows_relatum_with_metadata():
    permission = _make_primitive("permission", [
        _make_depth(DepthLevel.EXISTENCE),
        _make_depth(DepthLevel.IDENTITY),
        _make_depth(DepthLevel.CAPABILITIES),
        _make_depth(DepthLevel.CONSTRAINTS),
    ])
    relatum = Relatum(
        relation_type=RelationType.CONSTRAINED_BY,
        target_id=permission.id,
        target_depth=DepthLevel.CONSTRAINTS,
        metadata={"provenance": "authored"},
    )
    file_p = _make_primitive("file", [
        _make_depth(DepthLevel.EXISTENCE),
        _make_depth(DepthLevel.IDENTITY),
        _make_depth(DepthLevel.CAPABILITIES),
        _make_depth(DepthLevel.CONSTRAINTS, relata=[relatum]),
    ])
    result = _grounded(["file", "permission"], [file_p, permission])
    s = str(result)
    assert "→ permission  [CONSTRAINED_BY, target@D3]" in s
    assert "metadata: provenance=authored" in s


def test_grounding_str_shows_pathway():
    file_p = _make_primitive("file", [
        _make_depth(DepthLevel.EXISTENCE),
        _make_depth(DepthLevel.IDENTITY),
        _make_depth(DepthLevel.CAPABILITIES),
        _make_depth(DepthLevel.CONSTRAINTS),
    ])
    write_p = _make_primitive("write", [
        _make_depth(DepthLevel.EXISTENCE),
        _make_depth(DepthLevel.IDENTITY),
        _make_depth(DepthLevel.CAPABILITIES),
        _make_depth(DepthLevel.CONSTRAINTS),
    ])
    step = EpistemicStep(
        source_id=write_p.id,
        target_id=file_p.id,
        relation_type=RelationType.APPLIES_TO,
        source_depth=DepthLevel.CAPABILITIES,
        target_depth=DepthLevel.CAPABILITIES,
    )
    result = _grounded(["file", "write"], [file_p, write_p], pathway=[step])
    s = str(result)
    assert "Pathway:" in s
    assert "write —[APPLIES_TO@D2]→ file" in s


def test_grounding_str_deduplicates_pathway():
    file_p = _make_primitive("file")
    write_p = _make_primitive("write")
    step = EpistemicStep(
        source_id=write_p.id,
        target_id=file_p.id,
        relation_type=RelationType.APPLIES_TO,
        source_depth=DepthLevel.CAPABILITIES,
        target_depth=DepthLevel.CAPABILITIES,
    )
    result = _grounded(["file", "write"], [file_p, write_p], pathway=[step, step])
    s = str(result)
    assert s.count("write —[APPLIES_TO@D2]→ file") == 1


def test_grounding_str_not_grounded_header():
    result = GroundingResult(grounded=False, resolved=["api"], gaps=[], trace=None)
    s = str(result)
    assert s.startswith("[VRE] Not grounded — api")
    assert "Cannot execute until knowledge gaps are resolved." in s
    assert "This is your epistemic trace" not in s


def test_grounding_str_existence_gap():
    p = _make_primitive("api")
    gap = ExistenceGap(primitive=p)
    result = GroundingResult(grounded=False, resolved=["api"], gaps=[gap], trace=None)
    s = str(result)
    assert "EXISTENCE: 'api' is not in the knowledge graph" in s


def test_grounding_str_depth_gap_with_current():
    p = _make_primitive("file")
    gap = DepthGap(
        primitive=p,
        required_depth=DepthLevel.CONSTRAINTS,
        current_depth=DepthLevel.IDENTITY,
    )
    result = GroundingResult(grounded=False, resolved=["file"], gaps=[gap], trace=None)
    s = str(result)
    assert "DEPTH: 'file' known to D1 IDENTITY, requires D3 CONSTRAINTS" in s


def test_grounding_str_depth_gap_no_current():
    p = _make_primitive("file")
    gap = DepthGap(
        primitive=p,
        required_depth=DepthLevel.CONSTRAINTS,
        current_depth=None,
    )
    result = GroundingResult(grounded=False, resolved=["file"], gaps=[gap], trace=None)
    s = str(result)
    assert "DEPTH: 'file' known to none, requires D3 CONSTRAINTS" in s


def test_grounding_str_relational_gap():
    src = _make_primitive("file")
    tgt = _make_primitive("permission")
    gap = RelationalGap(
        source=src,
        target=tgt,
        required_depth=DepthLevel.CONSTRAINTS,
        current_depth=DepthLevel.IDENTITY,
    )
    result = GroundingResult(
        grounded=False, resolved=["file", "permission"], gaps=[gap], trace=None
    )
    s = str(result)
    assert "RELATIONAL: 'file' → 'permission' requires D3 CONSTRAINTS on target, found D1 IDENTITY" in s


def test_grounding_str_reachability_gap():
    p = _make_primitive("network")
    gap = ReachabilityGap(primitive=p)
    result = GroundingResult(
        grounded=False, resolved=["file", "network"], gaps=[gap], trace=None
    )
    s = str(result)
    assert "REACHABILITY: 'network' is not connected to other concepts" in s


def test_grounding_str_no_trace_renders_gracefully():
    result = GroundingResult(grounded=True, resolved=["file"], gaps=[], trace=None)
    s = str(result)
    assert "[VRE] Grounded — file" in s
    assert "═══" not in s


def test_grounding_str_compact_depth_format():
    """Primitive with all-empty depths renders as a single compact line."""
    p = _make_primitive("write", [
        _make_depth(DepthLevel.EXISTENCE),
        _make_depth(DepthLevel.IDENTITY),
        _make_depth(DepthLevel.CAPABILITIES),
        _make_depth(DepthLevel.CONSTRAINTS),
    ])
    result = _grounded(["write"], [p])
    s = str(result)
    assert "D0 EXISTENCE → D1 IDENTITY → D2 CAPABILITIES → D3 CONSTRAINTS" in s


def test_grounding_str_primitive_with_no_depths():
    """Transient primitive (no depths) renders just the header."""
    p = _make_primitive("unknown")
    trace = _make_trace([p])
    result = GroundingResult(grounded=False, resolved=["unknown"], gaps=[], trace=trace)
    s = str(result)
    assert "═══ unknown" in s
    # No depth content below the header
    assert "D0" not in s


# ── PolicyResult.__str__ ──────────────────────────────────────────────────────

def test_policy_str_pass():
    result = PolicyResult(action="PASS")
    assert str(result) == "[VRE Policy] PASSED"


def test_policy_str_pending():
    result = PolicyResult(action="PENDING", confirmation_message="Confirm this action?")
    assert str(result) == "[VRE Policy] PENDING — Confirm this action?"


def test_policy_str_block():
    result = PolicyResult(action="BLOCK", reason="Forbidden operation")
    assert str(result) == "[VRE Policy] BLOCKED — Forbidden operation"
