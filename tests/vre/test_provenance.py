"""
Unit tests for Provenance as a first-class attribute on Primitive, Depth, and Relatum.
"""

import json
import re
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vre.core.models import (
    Depth,
    DepthLevel,
    Primitive,
    Provenance,
    ProvenanceSource,
    Relatum,
    RelationType,
)
from vre.core.backends.sqlite import SQLiteRepository
from vre.core.errors import HydrationError, ProvenanceError
from vre.core.grounding.models import _fmt_depth, _fmt_primitive, _fmt_relatum


# ── Model construction ───────────────────────────────────────────────────────

def test_provenance_defaults():
    """
    Provenance constructed with only source gets auto-set timestamps and None detail.
    """
    p = Provenance(source=ProvenanceSource.AUTHORED)
    assert p.source == ProvenanceSource.AUTHORED
    assert isinstance(p.created_at, datetime)
    assert isinstance(p.updated_at, datetime)
    assert p.detail is None


def test_provenance_enum_values():
    """
    ProvenanceSource enum maps to the canonical string values from CLAUDE.md §7.2:
    two human-attested genealogies (authored, learned) plus one system genealogy
    (synthetic) for engine-generated, non-knowledge placeholders.
    """
    assert ProvenanceSource.AUTHORED.value == "authored"
    assert ProvenanceSource.LEARNED.value == "learned"
    assert ProvenanceSource.SYNTHETIC.value == "synthetic"
    assert [s.value for s in ProvenanceSource] == ["authored", "learned", "synthetic"]


def test_provenance_with_detail():
    """
    The optional detail field stores a human-readable context string.
    """
    p = Provenance(source=ProvenanceSource.LEARNED, detail="discovered via permission denied")
    assert p.detail == "discovered via permission denied"


def test_provenance_timestamps_auto_set():
    """
    Both created_at and updated_at are bracketed by the wall clock at construction time.
    """
    before = datetime.now(timezone.utc)
    p = Provenance(source=ProvenanceSource.AUTHORED)
    after = datetime.now(timezone.utc)
    assert before <= p.created_at <= after
    assert before <= p.updated_at <= after


def test_primitive_requires_provenance():
    """
    Primitive constructed without provenance is rejected at the model level —
    the contract is explicit and unavoidable, not optional (#98).
    """
    with pytest.raises(ValidationError):
        Primitive(name="test")


def test_depth_requires_provenance():
    """
    Depth constructed without provenance is rejected at the model level (#98).
    """
    with pytest.raises(ValidationError):
        Depth(level=DepthLevel.EXISTENCE)


def test_relatum_requires_provenance():
    """
    Relatum constructed without provenance is rejected at the model level (#98).
    """
    with pytest.raises(ValidationError):
        Relatum(
            relation_type=RelationType.APPLIES_TO,
            target_id=uuid4(),
            target_depth=DepthLevel.CAPABILITIES,
        )


def test_primitive_with_provenance():
    """
    Primitive accepts and stores a Provenance instance.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    p = Primitive(name="file", provenance=prov)
    assert p.provenance is prov
    assert p.provenance.source == ProvenanceSource.AUTHORED


def test_depth_with_provenance():
    """
    Depth accepts and stores a Provenance instance.
    """
    prov = Provenance(source=ProvenanceSource.LEARNED)
    d = Depth(level=DepthLevel.CONSTRAINTS, provenance=prov)
    assert d.provenance is prov


def test_relatum_with_provenance():
    """
    Relatum accepts and stores a Provenance instance.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    r = Relatum(
        relation_type=RelationType.APPLIES_TO,
        target_id=uuid4(),
        target_depth=DepthLevel.CAPABILITIES,
        provenance=prov,
    )
    assert r.provenance is prov


# ── Persistence boundary validation ──────────────────────────────────────────
# The model now requires provenance at construction (above), so these guard the
# save-boundary defense against an integrator nulling/spoofing provenance *after*
# construction — Pydantic does not re-validate on assignment, so validate_provenance
# is the fails-closed check the backends run before any write (#98).

def test_validate_provenance_rejects_primitive_without_provenance():
    """
    validate_provenance raises ProvenanceError when the primitive's provenance
    is nulled after construction.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    p = Primitive(name="test", provenance=prov)
    p.provenance = None
    with pytest.raises(ProvenanceError, match=re.escape("Primitive 'test' is missing provenance")):
        p.validate_provenance()


def test_validate_provenance_rejects_depth_without_provenance():
    """
    validate_provenance raises ProvenanceError when a depth's provenance is
    nulled after construction.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    p = Primitive(
        name="file",
        provenance=prov,
        depths=[Depth(level=DepthLevel.EXISTENCE, provenance=prov)],
    )
    p.depths[0].provenance = None
    with pytest.raises(ProvenanceError, match="depth D0 .EXISTENCE. is missing provenance"):
        p.validate_provenance()


def test_validate_provenance_rejects_relatum_without_provenance():
    """
    validate_provenance raises ProvenanceError when a relatum's provenance is
    nulled after construction.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    target_id = uuid4()
    p = Primitive(
        name="read",
        provenance=prov,
        depths=[
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=prov,
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=target_id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=prov,
                    ),
                ],
            ),
        ],
    )
    p.depths[0].relata[0].provenance = None
    with pytest.raises(ProvenanceError, match="relatum APPLIES_TO"):
        p.validate_provenance()


def test_validate_provenance_accepts_full_provenance():
    """
    validate_provenance succeeds when provenance is set at all levels.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    p = Primitive(
        name="file",
        provenance=prov,
        depths=[
            Depth(
                level=DepthLevel.EXISTENCE,
                provenance=prov,
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=prov,
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=uuid4(),
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=prov,
                    ),
                ],
            ),
        ],
    )
    p.validate_provenance()  # should not raise


# ── Round-trip serialization ─────────────────────────────────────────────────

def test_depths_to_json_includes_provenance():
    """
    _depths_to_json serializes depth provenance into the JSON entry.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    depths = [Depth(level=DepthLevel.EXISTENCE, provenance=prov)]
    result = json.loads(SQLiteRepository._depths_to_json(depths))
    assert len(result) == 1
    assert "provenance" in result[0]
    assert result[0]["provenance"]["source"] == "authored"


def test_hydrate_primitive_with_provenance():
    """
    Full round-trip: provenance at node, depth, and relatum levels survives serialize → hydrate.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    target_id = uuid4()
    primitive = Primitive(
        name="file",
        provenance=prov,
        depths=[
            Depth(
                level=DepthLevel.EXISTENCE,
                provenance=prov,
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=Provenance(source=ProvenanceSource.LEARNED, detail="from failure"),
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=target_id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=Provenance(source=ProvenanceSource.LEARNED),
                    ),
                ],
            ),
        ],
    )

    # Serialize
    depths_json = SQLiteRepository._depths_to_json(primitive.depths)
    node_prov_json = json.dumps(prov.model_dump(mode="json"))
    rel_prov = Provenance(source=ProvenanceSource.LEARNED)
    rel_prov_json = json.dumps(rel_prov.model_dump(mode="json"))

    node_data = {
        "id": str(primitive.id),
        "name": "file",
        "depths_json": depths_json,
        "provenance_json": node_prov_json,
        "metrics_json": None,
    }
    relationships = [
        {
            "rel_type": "APPLIES_TO",
            "target_id": str(target_id),
            "source_depth": 2,
            "target_depth": 2,
            "metadata_json": "{}",
            "provenance_json": rel_prov_json,
        },
    ]

    # Deserialize
    hydrated = SQLiteRepository._hydrate_primitive(node_data, relationships)

    # Node provenance
    assert hydrated.provenance is not None
    assert hydrated.provenance.source == ProvenanceSource.AUTHORED

    # Depth provenance
    d0 = next(d for d in hydrated.depths if d.level == DepthLevel.EXISTENCE)
    assert d0.provenance is not None
    assert d0.provenance.source == ProvenanceSource.AUTHORED

    d2 = next(d for d in hydrated.depths if d.level == DepthLevel.CAPABILITIES)
    assert d2.provenance is not None
    assert d2.provenance.source == ProvenanceSource.LEARNED
    assert d2.provenance.detail == "from failure"

    # Relatum provenance
    assert len(d2.relata) == 1
    assert d2.relata[0].provenance is not None
    assert d2.relata[0].provenance.source == ProvenanceSource.LEARNED


def test_hydrate_primitive_without_provenance_fails_closed():
    """
    Hydrating a legacy/unstamped node (no provenance) raises HydrationError —
    provenance is required, so such graphs are read-only-fail, not silently None (#98).
    """
    node_data = {
        "id": str(uuid4()),
        "name": "legacy",
        "depths_json": json.dumps([{"level": 0, "properties": {}}]),
        "provenance_json": None,
        "metrics_json": None,
    }
    with pytest.raises(HydrationError, match="missing provenance"):
        SQLiteRepository._hydrate_primitive(node_data, [])


# ── Display formatting ───────────────────────────────────────────────────────

def test_fmt_relatum_with_provenance():
    """
    _fmt_relatum includes a 'provenance: source (date)' line when provenance is set.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    target_id = uuid4()
    r = Relatum(
        relation_type=RelationType.APPLIES_TO,
        target_id=target_id,
        target_depth=DepthLevel.CAPABILITIES,
        provenance=prov,
    )
    lines = _fmt_relatum(r, {target_id: "file"})
    joined = "\n".join(lines)
    date_str = prov.created_at.strftime("%Y-%m-%d")
    assert f"provenance: authored ({date_str})" in joined


def test_fmt_depth_with_provenance():
    """
    _fmt_depth appends an inline [source] tag when provenance is set.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    d = Depth(level=DepthLevel.CAPABILITIES, provenance=prov)
    lines = _fmt_depth(d, {})
    assert lines[0] == "  D2 CAPABILITIES  [authored]"


def test_fmt_primitive_with_provenance():
    """
    _fmt_primitive includes a 'provenance: source (date)' line below the header.
    """
    prov = Provenance(source=ProvenanceSource.AUTHORED)
    p = Primitive(name="file", provenance=prov)
    lines = _fmt_primitive(p, {})
    joined = "\n".join(lines)
    date_str = prov.created_at.strftime("%Y-%m-%d")
    assert f"provenance: authored ({date_str})" in joined


