<img width="2048" height="2048" alt="vre_logo" src="https://github.com/user-attachments/assets/9419c6a6-4be6-418f-8199-090bdf5437a9" />


# VRE — Volute Reasoning Engine

## Note
This library is still in active development and is subject to breaking changes.
However, a stable v1.0.0 release is expected soon.

**Epistemic enforcement for autonomous agents.**

VRE is a Python library that gives autonomous agents an explicit, inspectable model of what they know before they act.
It is not a permissions system, a rules engine, or a safety classifier. It is a mechanism for making an agent's
knowledge boundary a first-class object — one that can be queried, audited, and enforced at runtime.

  ---

## Table of Contents

- [The Problem](#the-problem)
- [How It Works](#how-it-works)
    - [The Epistemic Graph](#the-epistemic-graph)
    - [Relata](#relata)
    - [Knowledge Gaps](#knowledge-gaps)
    - [Layered Safety](#layered-safety)
- [Scope](#scope)
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Infrastructure](#infrastructure)
    - [Seeding the Graph](#seeding-the-graph)
- [Core API](#core-api)
    - [Connecting to VRE](#connecting-to-vre)
    - [Agent Identity](#agent-identity)
    - [Checking Grounding](#checking-grounding)
    - [Using the Trace as Agent Context](#using-the-trace-as-agent-context)
    - [Checking Policy](#checking-policy)
- [The `vre_guard` Decorator](#the-vre_guard-decorator)
    - [Parameters](#parameters)
    - [Execution Flow](#execution-flow)
- [Callbacks](#callbacks)
    - [`on_trace`](#on_trace)
    - [`on_policy`](#on_policy)
- [Learning](#learning)
    - [How It Works](#how-it-works-1)
    - [Candidate Types](#candidate-types)
    - [Provenance](#provenance)
    - [Reachability Prerequisites](#reachability-prerequisites)
    - [Reference Loop](#reference-loop)
- [Policy System](#policy-system)
    - [Declaring a Policy](#declaring-a-policy)
    - [Policy Callbacks](#policy-callbacks)
    - [Evaluation Flow](#evaluation-flow)
    - [Registration & Validation](#registration--validation)
- [Integrations](#integrations)
- [Future](#future)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## The Problem

Modern LLM-based agents fail in a specific and consistent way: they act as if they know more than they can justify.

This is not a capability problem. The models are capable. It is an *epistemic* problem — the agent has no internal
representation of the boundary between what it genuinely understands and what it is
confabulating. Hallucination, unsafe execution, and overconfident planning are all symptoms of the same root cause: **epistemic opacity**.

When an agent is asked to delete files, migrate a database, or execute a shell command, the question is not only "can I
do this?" but "do I actually understand what I am doing well enough to do it
safely?" Current systems have no mechanism to answer that second question. They proceed anyway.

This is not hypothetical. In December 2025, Amazon's Kiro agent — given operator-level access to fix a small issue in
AWS Cost Explorer — decided the correct approach was to delete and recreate the
environment entirely, causing
a [13-hour outage](https://www.theregister.com/2026/02/20/amazon_denies_kiro_agentic_ai_behind_outage/). In February
2026, [OpenClaw deleted the
inbox](https://techcrunch.com/2026/02/23/a-meta-ai-security-researcher-said-an-openclaw-agent-ran-amok-on-her-inbox/) of
Summer Yue — Meta's Director of AI Alignment — after context window compaction
silently discarded her instruction to wait for approval before taking action. In each case, the agent acted confidently
on knowledge it could not justify. The safety constraints were linguistic —
instructions that could be forgotten, overridden, or reasoned around. VRE's constraints are structural.

VRE addresses this directly. It imposes a contract: before an action executes, the agent must demonstrate that the
relevant concepts are grounded in the knowledge graph at the depth required for
execution. If they are not, the action is blocked and the gap is surfaced explicitly. The agent does not guess. It does
not proceed on partial knowledge. It is structurally incapable of executing an
action it does not understand with respect to its epistemic model — and perhaps more importantly, it surfaces what it
does not know. Absence of knowledge is treated as a first-class object.

<img width="3168" height="710" alt="image" src="https://github.com/user-attachments/assets/4fedf455-a5d2-4443-acb5-ba85ac99f15c" />

---

## How It Works

### The Epistemic Graph

VRE maintains a graph of **primitives** — conceptual entities like `file`, `create`, `permission`, `directory`. These
are not tools or commands. They are concepts: the things an agent reasons *about*, not the mechanisms it uses to act.

Each primitive is grounded across a hierarchy of **depth levels**:

| Depth | Name         | Question answered                       |
|-------|--------------|-----------------------------------------|
| D0    | EXISTENCE    | Does this concept exist?                |
| D1    | IDENTITY     | What is it, in principle?               |
| D2    | CAPABILITIES | What can happen to it / what can it do? |
| D3    | CONSTRAINTS  | Under what conditions does that hold?   |
| D4+   | IMPLICATIONS | What follows if it happens?             |

Depth is **monotonic**: D3 grounding implies D0–D2 are also grounded. Depth requirements are derived from the graph
structure itself — edges carry a source depth that determines when they become
visible and a target depth that determines when they resolve. An integrator can also enforce a minimum depth floor (e.g.
D3 for execution) as a secondary safety lever.

### Relata

Primitives are connected by typed, directional, depth-aware **relata**:

```
create --[APPLIES_TO @ D2]--> file
file   --[CONSTRAINED_BY @ D3]--> permission
```

A relatum declares that understanding one concept at a given depth requires understanding another concept at a specified
depth. When VRE resolves a grounding query, it follows these dependencies and checks that the entire connected subgraph
meets the required depth. A relational gap — where a dependency's target is not grounded deeply enough — is surfaced as
a distinct gap type.

### Knowledge Gaps

When grounding fails, VRE returns structured gap objects — not generic errors. There are four gap types:

| Type              | Meaning                                                          |
|-------------------|------------------------------------------------------------------|
| `ExistenceGap`    | The concept is not in the graph at all                           |
| `DepthGap`        | The concept exists but is not grounded to the required depth     |
| `RelationalGap`   | A relatum's target does not meet the depth required by that edge |
| `ReachabilityGap` | The concept is not connected to the other submitted concepts     |

Gaps are not failures to be hidden. They are information. An existence gap on `network` tells you the agent has no
epistemic model of networking — not that the request was malformed. The agent can
surface this gap to the user, initiate a learning flow, or escalate to a human.

VRE does not require a complete or richly-detailed graph to be useful. The enforcement mechanism is structural — depth
requirements are derived from edge placement. A minimal graph with a handful of
primitives enforces the contract correctly. A richer graph adds better context, not stronger enforcement.

### Layered Safety

VRE is one layer of a deliberately layered safety model:

1. **Epistemic safety (VRE)** — prevents unjustified action. The agent cannot act on what it does not understand.
2. **Mechanical safety (tool constraints)** — constrains *how* the agent can act. Sandboxing, path restrictions,
   resource guards.
3. **Human safety (policy gates)** — requires explicit consent for elevated or destructive actions.

VRE governs only the first layer, by design. It does not replace sandboxing. It does not replace human oversight. It
makes those layers more meaningful by ensuring the agent understood what it was
doing when it asked for permission to act.

---

## Scope

**VRE is not a sandbox.** It does not isolate processes, restrict filesystem access, or enforce OS-level permissions. It
operates at the epistemic layer — determining whether an action is justified,
not whether it is physically permitted.

**VRE is not a safety classifier.** It does not scan outputs for harmful content or filter model responses. It gates
execution, not generation.

**VRE is not a replacement for human oversight.** Its policy gates are a mechanism for human oversight — surfacing
decisions that require consent and blocking until consent is given.

---

## Getting Started

### Installation

```bash
pip install vre
# or with Poetry
poetry add vre
```

VRE ships with a **SQLite backend** that works out of the box — no external services required.
The database defaults to `~/.vre/graph.db` and is created automatically on first use.

For production deployments or larger graphs, an optional **Neo4j backend** is available:

```bash
pip install vre[neo4j]
```

### Infrastructure

**SQLite (default)** — no setup needed. The database file is created automatically.

**Neo4j (optional)** — requires a running Neo4j instance:

```bash
docker run -d \
--name neo4j \
-p 7474:7474 -p 7687:7687 \
-e NEO4J_AUTH=neo4j/password \
neo4j:latest
```

### Seeding the Graph

The VRE repository ships with domain seeders in [`seeders/`](seeders/) and a
gap-demonstration script in [`scripts/`](scripts/). Seeders upsert primitives
by name (idempotent re-runs); the demo script clears the graph first to
produce its deterministic output. Use `scripts/clear_graph.py` if you want
a clean slate before seeding. See [`scripts/README.md`](scripts/README.md)
for details.

All scripts default to the SQLite backend. Pass `--backend neo4j` with
connection flags to use Neo4j instead.

```bash
# Fully grounded filesystem domain — 20 primitives, all at D3+ with complete relata
python seeders/seed_filesystem.py

# Gap demonstration graph — 10 primitives, deliberately shaped to produce each gap type
python scripts/seed_gaps.py

# Same commands with Neo4j:
python seeders/seed_filesystem.py \
    --backend neo4j --neo4j-uri neo4j://localhost:7687 --neo4j-user neo4j --neo4j-password password
```

---

## Core API

### Connecting to VRE

```python
from vre import VRE, SQLiteRepository

repo = SQLiteRepository()  # defaults to ~/.vre/graph.db
vre = VRE(repo)
```

Or with Neo4j:

```python
from vre import VRE
from vre.core.backends import Neo4jRepository

repo = Neo4jRepository(
    uri="neo4j://localhost:7687",
    user="neo4j",
    password="password",
)
vre = VRE(repo)
```

### Agent Identity

An optional `agent_key` associates the VRE instance with a stable agent identity. The key is resolved via a file-based
registry (`~/.vre/agents.json`) so that the same key always maps to the same UUID,
even across restarts. When configured, every `GroundingResult` carries the agent's `agent_id`.

```python
vre = VRE(repo, agent_key="my-agent", agent_name="My Agent")

vre.identity.agent_id  # stable UUID, persisted across restarts
vre.identity.name  # "My Agent"
```

`agent_name` is a human-readable label used only on first registration — subsequent calls with the same key return the
existing identity. Both parameters are optional; without `agent_key`, traces are
anonymous and `vre.identity` is `None`. You may also pass `registry_path` to customize the registry file location (
default: `~/.vre/agents.json`).

### Checking Grounding

```python
result = vre.check(["create", "file"])

print(result.grounded)  # True / False
print(result.resolved)  # ["create", "file"] — canonical names after resolution
print(result.gaps)  # [] or list of KnowledgeGap instances
print(result)  # Full formatted epistemic trace
```

`vre.check()` derives depth requirements from graph structure — edges at higher source depths are only visible when the
source primitive is grounded to that depth. An optional `min_depth` parameter
lets integrators enforce a stricter floor (e.g. D3 for execution). If any concept is unknown, lacks the required depth,
has an unmet relational dependency, or is disconnected from the other submitted
concepts, `grounded` is `False` and the corresponding gaps are surfaced.

### Using the Trace as Agent Context

`vre.check()` can be called before an agent runs to pre-load the epistemic trace into the model's context window. Rather
than letting the LLM reason from general knowledge alone, you give it the
graph's structured understanding of the relevant concepts before it decides what to do.

```python
result = vre.check(["delete", "file"])

if result.grounded:
    context = str(result)  # full structured trace, formatted for readability
    response = llm.invoke([
        SystemMessage(content="You are a filesystem agent."),
        SystemMessage(content=f"Epistemic context:\n{context}"),
        HumanMessage(content=user_input),
    ])
else:
    for gap in result.gaps:
        print(f"Knowledge gap: {gap}")
```

This is particularly useful for planning-mode interactions: the agent receives structured knowledge of what it
understands (and at what depth) before it proposes an action.

### Checking Policy

```python
policy = vre.check_policy(["delete", "file"], cardinality="multiple")

if policy.action == "BLOCK":
    print(policy.reason)
    for v in policy.violations:
        print(f"  - {v.message}")
```

Called without a `tool_call` (as here), a callback-bearing policy can't be evaluated, so the gate **fails closed** and
the policy fires — pass `tool_call=ToolCallContext(...)` (as `vre_guard` does) when you want the callback consulted.

`cardinality` hints whether the operation targets a single entity (`"single"`) or many (`"multiple"`, e.g. recursive or
glob). An optional `on_policy` callback handles violations that require human
confirmation — it receives only the confirmation-required violations and returns `True` to proceed or `False` to block.

---

## The `vre_guard` Decorator

`vre_guard` is the primary integration point. It wraps any callable and gates it behind a grounding check and a policy
evaluation before the function body executes. This is designed to wrap the tools
your agent uses to act on the world, ensuring that every action is epistemically justified and compliant with your
defined policies.

```python
from vre.guard import vre_guard

@vre_guard(vre, concepts=["write", "file"])
def write_file(path: str, content: str) -> str:
    ...
```

### Parameters

```python
vre_guard(
    vre,              # VRE instance
    concepts,         # list[str] or Callable(*args, **kwargs) -> list[str]
    cardinality=None, # str | None or Callable(*args, **kwargs) -> str | None
    min_depth=None,   # DepthLevel | None — enforces a minimum depth floor
    on_trace=None,    # Callable[[GroundingResult], None]
    on_policy=None,   # Callable[[list[PolicyViolation]], bool]
)
```

The guard does not orchestrate learning. When grounding fails, it returns the
`GroundingResult` and lets the integrator decide what to do next — typically
by exposing a separate `learn_gaps` tool that the agent can invoke. See
[Learning](#learning).

**`concepts`** can be static or dynamic. Static is appropriate when a function always touches the same concept domain.
Dynamic is appropriate when the concepts depend on the actual arguments — for
example, a shell tool that must inspect the command string:

```python
concepts = extract_concepts  # your callable: maps the command string to primitives

@vre_guard(vre, concepts=concepts)
def shell_tool(command: str) -> str:
    ...
 ```

VRE does not own concept extraction. The integrator decides how to map tool arguments to primitives — an LLM call, a
static alias table, a rule engine, or any combination.

**`cardinality`** can also be static or dynamic. When dynamic, it receives the same arguments as the decorated function:

```python
def get_cardinality(command: str) -> str:
    flags = {"-r", "-R", "-rf", "--recursive"}
    tokens = set(command.split())
    has_glob = any("*" in t for t in tokens)
    return "multiple" if (flags & tokens or has_glob) else "single"

@vre_guard(vre, concepts=concepts, cardinality=get_cardinality)
def shell_tool(command: str) -> str:
    ...
 ```

### Execution Flow

Each call runs the following sequence:

1. **Resolve concepts** — map names to canonical primitives via the graph
2. **Ground** — verify the subgraph meets depth requirements (graph-derived + optional `min_depth` floor)
3. **Fire `on_trace`** — surface the epistemic result to the caller
4. **If not grounded** — return the `GroundingResult` immediately; the function does not execute
5. **Evaluate policies** — check all `APPLIES_TO` relata for applicable policy gates
6. **If hard blocks** — return `PolicyResult(BLOCK)` immediately; `on_policy` is not consulted
7. **If confirmation required** — call `on_policy` with pending violations; block if declined or no handler
8. **If BLOCK** — return the `PolicyResult`; the function does not execute
9. **Execute** — call the original function and return its result

---

## Callbacks

### `on_trace`

Called after grounding, whether grounded or not. Receives the full `GroundingResult`. Use this to render the epistemic
trace to your UI.

```python
def on_trace(grounding: GroundingResult) -> None:
    if grounding.grounded:
        print(f"Grounded: {grounding.resolved}")
    else:
        for gap in grounding.gaps:
            print(f"Gap: {gap}")
```

`GroundingResult` carries:

- `grounded: bool` — whether all concepts are grounded with no gaps
- `resolved: list[str]` — canonical primitive names (or original if unresolvable)
- `gaps: list[KnowledgeGap]` — structured gap descriptions (`ExistenceGap`, `DepthGap`, `RelationalGap`,
  `ReachabilityGap`)
- `trace: EpistemicResponse | None` — the full subgraph with all primitives, depths, relata, and pathway
- `agent_id: UUID | None` — the stable agent identifier, when the VRE instance was created with an `agent_key`

For convenience, `result.get_primitives()` and `result.get_pathway_steps()` return the trace's primitives
and pathway steps directly (or empty lists when no trace is present), so callers don't have to drill into
`result.trace.result.*` themselves.

The reference integration renders `on_trace` as a Rich tree:

```
VRE Epistemic Check
├── ◈ create   ● ● ● ●
│   ├── APPLIES_TO  →  file       (target D2)
│   └── REQUIRES    →  filesystem (target D3)
├── ◈ file   ● ● ● ●
│   └── CONSTRAINED_BY  →  permission  (target D3)
└── ✓ Grounded — EPISTEMIC PERMISSION GRANTED
```

<img width="2786" height="1462" alt="image" src="https://github.com/user-attachments/assets/91d2ba34-716a-4d70-8c15-148a11e6c2b7" />

### `on_policy`

Called when policy evaluation produces violations that require human confirmation (`requires_confirmation=True`). Hard
blocks (`requires_confirmation=False`) are handled before `on_policy` is ever
consulted. Returns `True` to proceed, `False` to block.

```python
from vre.core.policy.models import PolicyViolation


def on_policy(violations: list[PolicyViolation]) -> bool:
    for v in violations:
        answer = input(f"Policy gate: {v.message} [y/N]: ").strip().lower()
        if answer != "y":
            return False
    return True
```

If `on_policy` is not provided and a policy requires confirmation, the guard returns
`PolicyResult(action=PolicyAction.BLOCK)` and the function does not execute.

<img width="1392" height="714" alt="image" src="https://github.com/user-attachments/assets/8b701635-d4ca-4511-98e3-cda82a5dde38" />


---

## Learning

VRE is a **knowledge linter**, not a knowledge builder. It identifies gaps and validates fills; the integrator
owns the loop. When grounding fails, the integrator decides whether to surface the gaps to the user, escalate
to a human, or run a learning loop that grows the graph through use.

This separation is deliberate. Loop orchestration is inherently integration-specific — different LLMs, different
data sources, different retry/budget strategies. By keeping VRE's surface tight (identify gaps, persist fills),
integrators can build whatever flow fits their stack without fighting the framework.

### How It Works

VRE exposes three things:

1. **`vre.check(concepts)`** returns a `GroundingResult` with structured `KnowledgeGap` objects when grounding fails
2. **`template_for_gap(gap)`** returns a candidate to fill. For depth and relational gaps it is **pre-seeded with the
   exact missing levels** (`gap.missing_levels` — the holes in `(current, required]` not already present), one empty
   slot each, so the integrator fills only the `properties`. VRE resolves *which* levels are missing; the integrator
   supplies *what they contain* (LLM structured output, user input, static rules)
3. **`vre.learning_engine.learn_gap(gap, candidate)`** validates the candidate against the **live** graph and persists
   it (always stamped `LEARNED`)

A typical integrator-owned loop looks like this:

```python
from vre.learning.templates import template_for_gap

grounding = vre.check(["delete", "file"])
while not grounding.grounded and grounding.gaps:
    gap = grounding.gaps[0]
    candidate = template_for_gap(gap)         # pre-seeded with gap.missing_levels
    filled = my_llm_fill(candidate, gap, grounding)  # fill the properties of each slot
    if filled is None:
        break
    vre.learning_engine.learn_gap(gap, filled)
    grounding = vre.check(["delete", "file"])
```

`learn_gap` raises `CandidateValidationError` if the candidate is malformed or its prerequisites are not met (e.g.
re-authoring an already-grounded level, or placing an edge at a depth the source does not have), and `GapResolvedError`
if the gap already closed underneath the candidate. The integrator catches the error, revises or re-grounds, and
retries.

### Candidate Types

Each gap type has a corresponding candidate model. Candidates carry only what's *new* — all context (primitive IDs,
existing depths, required depths) lives on the gap itself.

| Gap Type          | Candidate               | What the Integrator Fills In                                                  |
|-------------------|-------------------------|-------------------------------------------------------------------------------|
| `ExistenceGap`    | `ExistenceCandidate`    | D1 identity for a new concept (D0 is auto-generated)                          |
| `DepthGap`        | `DepthCandidate`        | Missing depth levels with properties                                          |
| `RelationalGap`   | `RelationalCandidate`   | Missing depth levels on the edge target                                       |
| `ReachabilityGap` | `ReachabilityCandidate` | Edge placement: source name, target name, relation type, source/target depths |

`ExistenceCandidate`, `DepthCandidate`, and `RelationalCandidate` all use `ProposedDepth`:

```python
from vre.learning.models import ProposedDepth

ProposedDepth(
    level=DepthLevel.CAPABILITIES,
    properties={"operations": ["read", "write"], "attributes": ["size", "permissions"]},
)
```

`ReachabilityCandidate` declares both source and target by name. At least one of them must match the gap's
primitive — the edge must fix *this* disconnection — but the integrator chooses the direction. An edge from an
existing connected node *back* to the orphan is just as valid as one originating from the orphan.

### Provenance

`learn_gap` accepts an optional `source: ProvenanceSource` parameter (default `LEARNED`). The integrator decides
how to stamp persisted knowledge based on its own loop semantics — `LEARNED` for agent-proposed fills approved at
the persistence boundary, `AUTHORED` for content a human drafted directly. Both are human-attested by construction;
provenance is genealogy, not a trust gradient. The graph remembers not just what it knows, but how it came to know it.

### Reachability Prerequisites

`ReachabilityCandidate` focuses solely on edge placement — it declares *where* the edge goes, not what depths need
to exist. If the source or target lacks the required depth level, `learn_gap` raises `CandidateValidationError`.

To handle this cleanly, the engine exposes `reachability_prerequisites(gap, candidate)` which returns a list of
`DepthGap` objects that must be filled before the edge can be placed. The integrator's loop checks prerequisites,
fills them, and only then calls `learn_gap` for the reachability candidate.

```python
prereqs = vre.learning_engine.reachability_prerequisites(gap, filled)
for depth_gap in prereqs:
    depth_filled = my_llm_fill(template_for_gap(depth_gap), depth_gap, grounding)
    vre.learning_engine.learn_gap(depth_gap, depth_filled)
vre.learning_engine.learn_gap(gap, filled)
```

### Reference Loop

The integrator owns the loop — VRE provides the pieces (`check`, `template_for_gap`, `learn_gap`) and you decide how
to drive them. A minimal loop grounds, fills each gap however you choose (LLM, human, or rules), persists, and
re-grounds until grounded:

```python
from vre.core.errors import CandidateValidationError, GapResolvedError
from vre.learning.templates import template_for_gap

grounding = vre.check(concepts)
while not grounding.grounded and grounding.gaps:
    gap = grounding.gaps[0]
    candidate = my_fill(template_for_gap(gap), gap, grounding)  # LLM / human / rules
    if candidate is None:
        break  # the operator deliberately chose to leave this gap open
    try:
        vre.learning_engine.learn_gap(gap, candidate)
    except GapResolvedError:
        pass  # the gap closed out from under us — just re-ground
    except CandidateValidationError as err:
        candidate = my_fill(template_for_gap(gap), gap, grounding, feedback=str(err))  # revise and retry
    grounding = vre.check(concepts)  # re-observe; the gap set shrinks
```

`learn_gap` is the privileged, human-gated persist call: agents *propose* candidates, they never get graph write
access. It validates each candidate against live graph state and raises `CandidateValidationError` (malformed or
out-of-scope) or `GapResolvedError` (the gap was already resolved). Reachability gaps additionally use
`reachability_prerequisites` (above) to fill missing depths before the edge is placed.

---

## Policy System

Policies gate specific concept relationships: which actions require confirmation, under
what cardinality conditions they fire, and what confirmation message to surface.

Policies are **code-resident**. You declare them in your own Python with the
`policy_callback` decorator, binding a callable to one `APPLIES_TO` edge
(`source_primitive` → `target_primitive` at a `source_depth`). The graph stores only
knowledge; policies are never persisted — so a tampered `~/.vre/graph.db` can neither
inject code nor re-point a callback. The gate only ever invokes callables your own
imported code handed it.

### Declaring a Policy

```python
from vre import policy_callback, DepthLevel, PolicyCallContext, PolicyCallbackResult


@policy_callback(
    source_primitive="delete",
    target_primitive="file",
    source_depth=DepthLevel.CONSTRAINTS,   # pins exactly one APPLIES_TO edge (D3 / execution)
    # key="..." identifies the policy; omitted here, so it defaults to the function name
    name="Protected file guard",
    requires_confirmation=False,           # hard block — no confirmation prompt
    confirmation_message="Deletion blocked by protected file policy.",
)
def protected_file(context: PolicyCallContext) -> PolicyCallbackResult:
    """Block deletion of files matching 'protected*'."""
    command = context.tool_call.call_args[0] if context.tool_call.call_args else ""
    targets = [t for t in command.split()[1:] if not t.startswith("-")]
    for target in targets:
        if target.startswith("protected"):
            return PolicyCallbackResult(passed=False, message=f"'{target}' is a protected file.")
    return PolicyCallbackResult(passed=True, message="No protected files affected.")
```

Importing the module that holds this declaration registers it. **Import your policy
modules before constructing `VRE`** — registration is an import-time side effect, and
`VRE(...)` validates every declared placement against the graph and then freezes the
registry (see *Registration & validation* below).

### Policy Callbacks

A `PolicyCallback` is a callable attached to a `Policy` that runs *during* evaluation to make domain-specific pass/fail
decisions. This is distinct from `on_policy`, which handles human confirmation
*after* violations are collected. A policy callback determines whether a violation fires at all.

The callback receives a `PolicyCallContext` composed of four parts: `tool_call` (the invocation —
`tool_name`, `call_args`, `call_kwargs`), `grounding` (a bounded facade — `agent_id` and the
`resolved_concepts` grounded in this call), `triggering_edge` (the specific edge that fired the
callback — source/target concept and the source/target depths), and `policy` (the `Policy` that
fired, including its `metadata`). It returns a `PolicyCallbackResult` — `passed=True` suppresses the
violation, `passed=False` fires it.

**Stateful callbacks** (instances that can't be decorated) use the imperative twin,
`register_policy`:

```python
from vre import register_policy, DepthLevel

register_policy(
    RateLimiter(per_minute=5),                # any callable taking a PolicyCallContext
    key="rate_limit",
    source_primitive="send", target_primitive="email",
    source_depth=DepthLevel.CONSTRAINTS, name="Rate limit",
)
```

**One callback, several edges** — stack decorators, each with a distinct key (the
decorator returns the original function, so it composes):

```python
@policy_callback(key="protected_file", source_primitive="delete",
                 target_primitive="file", source_depth=DepthLevel.CONSTRAINTS, name="Protected file")
@policy_callback(key="protected_dir", source_primitive="delete",
                 target_primitive="directory", source_depth=DepthLevel.CONSTRAINTS, name="Protected dir")
def protected_delete(context): ...
```

A callback can make nuanced, context-aware decisions by inspecting both the command arguments (via the
`ToolCallContext`) and external state — for example, an `rm` guard that resolves literal filenames, expands globs
against the filesystem, and inspects directories recursively before deciding whether a protected path is in scope.

### Evaluation Flow

1. **Cardinality filter** — if the policy specifies a `trigger_cardinality`, it only fires when the operation's
   cardinality matches
2. **Callback evaluation** — the callback runs with the full call context; `passed=True` suppresses the violation. The
   gate **fails closed** with a detailed reason (the policy fires) when the callback cannot be evaluated — no `tool_call`
   in context, or the callback raises — so a buggy callback never weakens a gate or escapes as a raw exception
3. **Violation collection** — unsuppressed policies produce `PolicyViolation` objects
4. **Hard blocks vs confirmation** — violations with `requires_confirmation=False` are immediate blocks. Those with
   `requires_confirmation=True` are deferred to the `on_policy` handler

### Registration & Validation

Declared policies live in a `PolicyRegistry`; the `policy_callback` decorator and `register_policy` write to a
module-global one that `VRE` reads by default (pass `policy_registry=` to use your own). At construction `VRE`:

- logs the registered policy keys — a `0 registered` line is your cue that a policy module was never imported;
- validates every declared placement against the graph. A placement whose `APPLIES_TO` edge is **absent** (typo,
  missing knowledge, or wrong depth) raises `PolicyPlacementError`, because a declared gate that protects nothing is the
  one dangerous, otherwise-silent case. Pass `validate_policies=False` to defer, then call
  `vre.validate_policy_placements()` yourself;
- freezes the registry, so the invariant *everything enforced was validated* holds.

The symmetry: a missing **callback** fails closed (the policy fires); a missing **edge** fails loud (`VRE` refuses to
start). Pass `expect_policies=N` to additionally assert the registered count.

**Multiple graphs in one process.** Freeze is per-registry, so give each graph its own `PolicyRegistry` and decorate
with *its* `.policy_callback`; constructing one VRE validates and freezes only that registry, never another graph's:

```python
from vre import VRE, PolicyRegistry, DepthLevel

reg_a = PolicyRegistry()

@reg_a.policy_callback(source_primitive="delete", target_primitive="file",
                       source_depth=DepthLevel.CONSTRAINTS, name="Protect A's files")
def protect_a(ctx): ...

vre_a = VRE(repo_a, policy_registry=reg_a)   # validates + enforces only reg_a, against graph A
vre_b = VRE(repo_b, policy_registry=reg_b)   # independent: reg_b, against graph B
```

The module-level `policy_callback` / `register_policy` are just this bound to a shared default registry — fine for the
common single-graph case.

**Validated at init ≠ guaranteed to fire.** Validation confirms the declared edge *exists* in the graph; it does not
guarantee the edge is *grounded* on a given call. `APPLIES_TO` is non-transitive, so grounding `["delete"]` alone
strips the `delete → file` relatum from the trace — the placement validates clean yet never fires, and `delete` can
fully ground and PASS. Enforcement still requires the action **and** the object concept in the query (the same static
conjunction the tool declares or the extractor emits). This isn't a regression — a graph-resident policy on a stripped
relatum behaved identically — but the init check verifies the edge is *reachable*, not that any given call will hit it.

---

## Integrations

VRE is framework-agnostic: `vre_guard` wraps any callable, and the learning loop is plain Python (see
[Learning](#learning)). A reference LangChain + Ollama agent previously lived under `examples/`; it has been retired —
it leaned on a framework that obscured where the guard sits and a `shell=True` pseudo-sandbox — in favor of a single
agent-driven showcase that teaches the concept-binding gradient end-to-end
([#114](https://github.com/anormang1992/vre/issues/114)).

---

## Future

### Learning Through Failure

When a mechanical failure occurs during execution — permission denied, missing dependency, invalid path — the failure
reveals a constraint that was not modeled. The agent proposes the missing relatum
(e.g. `create --[CONSTRAINED_BY]--> permission`), seeks human validation, and persists the new knowledge. Depth was
honest before the failure and more complete after.

### Knowledge Import

A pathway for growing an agent's graph from peer-published knowledge. An agent fetches a peer's subgraph for a target
concept and persists it locally as ordinary primitives stamped with `provenance.source = PEER` and a
`(peer_name, imported_at)` attestation. Imports are one-shot — refresh is an explicit operator action, never a live
link — which preserves the depth-explicit *validated trust* VRE's enforcement depends on while letting an agent grow
its graph from a community of peers instead of authoring every concept from scratch.

### Epistemic Memory

A new class of memory that stores not just information but the agent's epistemic relationship to that information.
Memories are indexed by concept and depth, decay or are reinforced based on usage and
grounding history, and affect the agent's confidence in related concepts.

---

## Tech Stack

| Concern            | Technology                                    |
|--------------------|-----------------------------------------------|
| Language           | Python 3.12+                                  |
| Epistemic graph    | SQLite (default) or Neo4j (`pip install vre[neo4j]`) |
| Concept resolution | Exact, case-insensitive name match (no NLP)   |
| Data models        | Pydantic v2                                   |
| Package management | Poetry                                        |


---

## Project Structure

```
src/vre/
├── __init__.py                  # VRE public interface (check, check_policy, learning_engine)
├── guard.py                     # vre_guard decorator (grounding → policy → execution)
├── metrics.py                   # MetricsManager — best-effort grounding metric updates
├── tracing.py                   # TraceWriter + TraceManager — JSONL persistence
│
├── identity/
│   ├── models.py                # AgentIdentity — stable UUID bound to a registration key
│   └── registry.py              # AgentRegistry — file-based, append-only identity persistence
│
├── core/
│   ├── models.py                # Primitive, Depth, Relatum, RelationType, DepthLevel, KnowledgeGap, Provenance
│   ├── errors.py                # VREError hierarchy — typed exceptions for all failure modes
│   ├── backends/
│   │   ├── repository.py        # Repository ABC — abstract persistence contract
│   │   ├── sqlite.py            # SQLiteRepository — SQLite backend (default)
│   │   └── neo4j.py             # Neo4jRepository — Neo4j backend (optional)
│   ├── grounding/
│   │   ├── engine.py            # GroundingEngine — depth-gated query, gap detection
│   │   └── models.py            # GroundingResult
│   └── policy/
│       ├── models.py            # Policy, Cardinality, PolicyResult, PolicyViolation
│       ├── registry.py          # PolicyRegistry, policy_callback, register_policy — code-resident policies
│       ├── gate.py              # PolicyGate — overlays registry placements onto a trace
│       └── callback.py          # PolicyCallContext, PolicyCallback protocol
│
└── learning/
    ├── models.py                # Candidate models with validate_for_gap methods
    ├── templates.py             # template_for_gap — gap → candidate model class
    └── engine.py                # LearningEngine — learn_gap, reachability_prerequisites

scripts/
├── clear_graph.py               # Clear all primitives from the graph
└── seed_gaps.py                 # Seed gap-demonstration graph (10 primitives)

seeders/
└── seed_filesystem.py           # Filesystem domain — 20 primitives, idempotent upsert
```

---

## Guiding Principle

> **The agent must never act as if it knows more than it can justify.**

VRE exists to enforce that rule — not as a policy, but as a structural property of the system.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your proposed changes. For major changes,
please discuss them in an issue first to ensure alignment with the project's
goals and architecture.

Areas where contributions would be particularly valuable:

- Additional seed scripts for more complex domains (e.g. networking, databases, cloud infrastructure)
- Integration examples with other Python agent frameworks or tool libraries — any integration submission should include
  a demo that exercises the integration and demonstrates epistemic resolution
  behavior
- VRE integration into other language environments (Node.js, Go, etc.)

This is a project that I am passionate about and is the culmination of almost 10 years of philosophical thought. I hope
to connect with other like-minded community members who prioritize safety and
epistemic integrity in autonomous agentic systems.

I look forward to seeing how this evolves!
