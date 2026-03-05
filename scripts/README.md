# Scripts

Utility scripts for managing the VRE Neo4j graph. All scripts accept
`--neo4j-uri`, `--neo4j-user`, and `--neo4j-password` arguments
(defaults: `neo4j://localhost:7687`, `neo4j`, `password`).

## clear_graph.py

Deletes every `Primitive` node and its relationships from the graph.
Called automatically by both seed scripts to ensure a clean slate, but
can also be run standalone.

```bash
poetry run python scripts/clear_graph.py
```

## seed_all.py

Seeds the fully grounded epistemic graph — 15 primitives covering the
filesystem domain end-to-end. Every primitive is grounded to D3
(CONSTRAINTS) with complete structural relata (DEPENDS_ON, REQUIRES,
INCLUDES, APPLIES_TO, CONSTRAINED_BY).

Primitives: `operating_system`, `filesystem`, `path`, `permission`,
`directory`, `file`, `user`, `group`, `create`, `read`, `write`,
`delete`, `list`, `move`, `copy`

```bash
poetry run python scripts/seed_all.py
```

## seed_gaps.py

Seeds a deliberately shaped graph (10 primitives) designed to
demonstrate depth-gated traversal and the full gap taxonomy. Each
action primitive is truncated or structured to produce a specific gap
type when queried.

```bash
poetry run python scripts/seed_gaps.py
```

### Primitives

| Primitive | Depths | Role |
|---|---|---|
| `operating_system` | D0–D3 | Fully grounded substrate |
| `filesystem` | D0–D3 | Fully grounded substrate |
| `path` | D0–D3 | Fully grounded structural |
| `permission` | D0–D3 | Fully grounded structural |
| `directory` | D0–D3 | Fully grounded entity — target for safe ops |
| `file` | D0–D1 | Shallow entity — triggers RelationalGap when targeted |
| `read` | D0–D2 | Safe action, APPLIES_TO@D2 visible |
| `list` | D0–D2 | Safe action, APPLIES_TO@D2 visible |
| `delete` | D0–D2 | Destructive action, APPLIES_TO@D3 omitted (D3 not seeded) |
| `create` | D0–D1 + D3 | Destructive action, APPLIES_TO@D3 gated (D2 missing) |

### Expected Gap Scenarios

| Query | Gaps | Why |
|---|---|---|
| `["list", "directory"]` | None | list@D2 sees APPLIES_TO@D2; directory@D3 >= target D2 |
| `["read", "file"]` | RelationalGap(read→file, req=D2, cur=D1) | read@D2 sees APPLIES_TO@D2, but file@D1 < target D2 |
| `["delete", "file"]` | ReachabilityGap(file) | APPLIES_TO lives at D3 in seed_all; D3 omitted here, no edges exist |
| `["delete", "directory"]` | ReachabilityGap(directory) | Same — no visible connection between delete and directory |
| `["create", "file"]` | DepthGap(create, req=D3, cur=D1) + ReachabilityGap(file) | create contiguous max=D1; APPLIES_TO@D3 gated, nodes disconnected |
| `["create", "directory"]` | DepthGap(create, req=D3, cur=D1) + ReachabilityGap(directory) | Same mechanism — edge gated, target unreachable |
| Unknown concept (e.g. `["frobnicate"]`) | ExistenceGap | Concept not in graph |

> Query with `min_depth=DepthLevel.CONSTRAINTS` to additionally produce
> DepthGap on any root primitive that doesn't reach D3 (read, list,
> delete, file).
