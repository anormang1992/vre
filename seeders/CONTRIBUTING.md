# Authoring a Seeder

This package is where contributors author **domain seed scripts** — modules
that upsert a connected set of primitives into a VRE Neo4j graph. Each module
is a self-contained domain (filesystem, HTTP, billing, etc.) that an
integrator can install standalone or alongside others.

Read this before adding a new seeder. The canonical reference implementation
is [`seed_filesystem.py`](./seed_filesystem.py) — copy its shape.

## Mental model: graph as policy

VRE doesn't enforce behavior with imperative rules — it enforces it with the
**shape of the graph**. Two axes do the policing:

1. **Depth coverage.** A primitive grounded only to D1 (IDENTITY) cannot
   answer questions that require D2 (CAPABILITIES) or deeper. Depth-gated
   traversal refuses to consult depths that haven't been authored. When a
   contributor stops at D1, the contributor has declared "this is all we
   know" — the engine respects that and surfaces a `DepthGap`.

2. **Relation placement.** Every relatum carries a `source_depth` (the depth
   on the source primitive where the edge lives) and a `target_depth` (the
   minimum depth the target must reach for the edge to be satisfied).
   - The edge is **invisible** until the source is grounded to `source_depth`.
   - The edge is **unsatisfied** if the target hasn't reached `target_depth`,
     producing a `RelationalGap`.

   Placing `Delete → File [APPLIES_TO]` at the source's D3 (CONSTRAINTS)
   instead of D2 (CAPABILITIES) means: "to know what delete acts on, you
   must first understand its constraints." That's policy, expressed as
   topology.

Seeders ship **epistemic facts**: what the domain knows, structured into
depths and connected via typed relations. Seeders do **not** ship
operational policy (`Policy(...)` objects belong to the integrator that
runs the application). The graph is the user's epistemic property; a
seeder describes a domain to it, no more.

## Depth levels (D0–D4)

| Depth | Name | Authoring intent |
|------:|------|------------------|
| D0 | EXISTENCE | The primitive exists. No properties; cheap to declare. |
| D1 | IDENTITY | What the thing **is** — attributes, defining properties, names of its parts. |
| D2 | CAPABILITIES | What the thing **can do** — operations, services, the verbs it supports. |
| D3 | CONSTRAINTS | What **bounds** it — preconditions, invariants, what restricts or governs the thing. |
| D4 | IMPLICATIONS | What **follows** from operating on it — cascading consequences, downstream epistemic effects. |

A primitive's depths must be **contiguous starting at D0**. You can stop at
any level (D0–D1 is fine for a placeholder entity), but you cannot have D0,
D1, D3 with D2 missing — the contiguous-max gates D3 to invisibility. Use
this deliberately: omitting D2 is how you say "we haven't grounded
capabilities yet."

Substrates and shared structurals (operating systems, filesystems,
permissions) typically need D0–D3. Actions usually need D0–D2 at minimum;
destructive or consequential actions extend to D3 (preconditions, required
permissions) and sometimes D4 (cascades). These are heuristics, not rules.

## Relation types

Relation types describe the **kind of epistemic connection** between two
primitives. Any relation can be authored at any depth — the depth is where
the *fact* is asserted, not a fixed slot for the relation type. The same
target may even appear at multiple depths via different relations.

| Relation | Transitive? | What it asserts |
|----------|:-----------:|-----------------|
| `REQUIRES` | yes | A hard prerequisite — the target must exist and be grounded for the source to be coherent. |
| `DEPENDS_ON` | yes | A runtime dependency — the source needs the target's services to function. |
| `CONSTRAINED_BY` | yes | Restriction — the source's behavior is bounded by the target. |
| `INCLUDES` | no | Composition — the source contains the target as a member. |
| `APPLIES_TO` | no | Action targeting — declares what an action operates on. |

**Transitive vs non-transitive.** Grounding resolution walks transitive
edges (`REQUIRES`, `DEPENDS_ON`, `CONSTRAINED_BY`) to compute the substrate
an action sits on. Non-transitive edges (`APPLIES_TO`, `INCLUDES`) define
one-step structural facts but do not contribute to the transitive substrate
— they are queried at the depth they live on, not followed recursively.

### Choosing a depth for an edge

Ask: **at what depth does this fact become true?** That's where the relatum
belongs. Some illustrative placements from the filesystem seeder:

- `Directory [D1] —REQUIRES→ Path [D1]`: a directory's *identity* is
  partly constituted by having a path. The fact lives at D1 because that's
  where directory becomes itself.
- `Directory [D2] —DEPENDS_ON→ Filesystem [D1]`: a directory's
  *capabilities* (create, list, delete) need a filesystem to operate
  against. The dependency is a runtime fact, asserted at D2.
- `Directory [D3] —CONSTRAINED_BY→ Permission [D1]`: a directory's
  *constraints* include permission checks. The fact lives at D3 because
  it's part of what bounds the directory's use.
- `Read [D2] —APPLIES_TO→ File [D2]`: read's targeting is known once its
  capabilities are grounded.
- `Delete [D3] —APPLIES_TO→ File [D2]`: delete's targeting is gated by
  understanding its constraints. Authoring this at D3 instead of D2 means
  the engine refuses to traverse the edge until delete is grounded
  through D3. Seeding delete at only D0–D2 → APPLIES_TO invisible →
  `ReachabilityGap` from delete to file (intentional).

The lesson: **the same relation type can sit at any depth**, and where it
sits is itself a policy decision. Use depth to declare what an agent must
understand before the connection is consultable.

## Idempotency contract

- Use `repo.upsert_primitive(prim)` — never `repo.save_primitive(prim)`.
  Upsert preserves the existing UUID when a primitive of the same name is
  already in the graph and logs an INFO message on overwrite.
- **Do not call `clear_graph`** from a seeder. Multiple domains may be
  installed in the same graph; clearing is the user's choice, not the
  seeder's. If a user wants a clean slate, they run
  `python scripts/clear_graph.py` first.
- Re-running a seeder is a no-op when the declared state matches the
  graph state, and an overwrite when it doesn't. Depths and relata are
  full-replaced for each upserted primitive — within-domain idempotency
  only. Cross-domain primitive merging is not handled by seeders; if two
  domains declare `file`, the second to run wins.

## What NOT to include

- **`Policy(...)` objects.** Operational policy (confirmation prompts,
  cardinality triggers, callbacks) belongs to the application that uses
  the graph, not to the seed.
- **Integrator-specific metadata.** Anything tied to a particular harness,
  agent framework, or runtime should not appear in seeded relata.
- **Ephemeral or runtime state.** Metrics, learned candidates, session
  data — these are written by the engine at runtime.
- **Redundant depth content.** Don't restate the IDENTITY description at
  D2. Each depth answers a different question.

## Authoring template

Structure the module **top-down**: substrates → structurals → entities →
actors → actions. Each `seed_*` function is defined above the functions
that depend on it. `main` lives at the bottom.

```python
"""
Seeder for the <domain> domain.

Upserts <one-sentence summary>. Idempotent — re-running preserves existing
primitive ids by name. Does not clear the graph.

Run: python seeders/seed_<domain>.py
"""
import argparse

from vre.core.graph import PrimitiveRepository
from vre.core.models import (
    Depth, DepthLevel, Primitive, Provenance, ProvenanceSource,
    Relatum, RelationType,
)

SEED_PROVENANCE = Provenance(source=ProvenanceSource.AUTHORED)


def seed_substrate(repo: PrimitiveRepository) -> Primitive:
    """Seed the foundational substrate at D0–D3."""
    substrate = Primitive(
        name="substrate",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={"description": "...", "attribute_a": "..."},
            ),
            # D2, D3 as needed
        ],
    )
    substrate = repo.upsert_primitive(substrate)
    print(f"Saved: substrate ({substrate.id})")
    return substrate


def seed_entity(repo: PrimitiveRepository, substrate: Primitive) -> Primitive:
    """Seed an entity that depends on the substrate."""
    entity = Primitive(
        name="entity",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={"description": "..."},
                relata=[
                    Relatum(
                        relation_type=RelationType.REQUIRES,
                        target_id=substrate.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                    ),
                ],
            ),
        ],
    )
    entity = repo.upsert_primitive(entity)
    print(f"Saved: entity ({entity.id})")
    return entity


def seed_action(repo: PrimitiveRepository, entity: Primitive) -> Primitive:
    """Seed an action whose APPLIES_TO targeting lives at D2."""
    action = Primitive(
        name="action",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={"description": "..."},
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={"description": "..."},
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=entity.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                    ),
                ],
            ),
        ],
    )
    action = repo.upsert_primitive(action)
    print(f"Saved: action ({action.id})")
    return action


def main(repository: PrimitiveRepository) -> None:
    with repository as repo:
        repo.ensure_constraints()
        substrate = seed_substrate(repo)
        entity = seed_entity(repo, substrate)
        seed_action(repo, entity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="<domain> domain seeder")
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    args = parser.parse_args()

    repo = PrimitiveRepository(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )
    main(repository=repo)
```

### Post-creation wiring

When two primitives reference each other (e.g., a filesystem `INCLUDES` a
file, but the file was seeded after the filesystem), seed both first, then
mutate one and re-upsert. See the post-creation block at the bottom of
`main()` in `seed_filesystem.py` for the pattern.

## Verifying a new seeder

1. Run the seeder against a local Neo4j and confirm no exceptions.
2. Re-run it. You should see `INFO  vre.core.graph  Upserting '<name>': overwriting existing primitive (id=...)` for every primitive — no duplicates created.
3. Use the grounding engine in a small script (or REPL) to query a representative concept from your domain. Inspect the resulting gaps and verify they match your authored shape — both clean passes and the gaps you intentionally left.
