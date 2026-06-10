<img width="2048" height="2048" alt="vre_logo" src="https://github.com/user-attachments/assets/9419c6a6-4be6-418f-8199-090bdf5437a9" />


# VRE — Volute Reasoning Engine

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
    - [Defining Policies](#defining-policies)
    - [Policy Callbacks](#policy-callbacks)
    - [Evaluation Flow](#evaluation-flow)
    - [Policy Wizard](#policy-wizard)
- [Integrations](#integrations)
    - [LangChain + Ollama Reference Agent](#langchain--ollama-reference-agent)
    - [Claude Code Hook](#claude-code-hook)
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
concepts = ConceptExtractor()  # LLM-based — see examples/langchain_ollama/callbacks.py

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
2. **`template_for_gap(gap)`** returns the candidate model class to fill — the integrator constructs an instance
   however they like (LLM structured output, user input, static rules)
3. **`vre.learning_engine.learn_gap(gap, candidate, source=LEARNED)`** validates the candidate against its gap and
   persists it to the graph

A typical integrator-owned loop looks like this:

```python
from vre.learning.templates import template_for_gap

grounding = vre.check(["delete", "file"])
while not grounding.grounded and grounding.gaps:
    gap = grounding.gaps[0]
    candidate_cls = template_for_gap(gap)
    filled = my_llm_fill(candidate_cls, gap, grounding)  # integrator's code
    if filled is None:
        break
    vre.learning_engine.learn_gap(gap, filled)
    grounding = vre.check(["delete", "file"])
```

`learn_gap` raises `CandidateValidationError` if the candidate is malformed or if its prerequisites are not met
(e.g. trying to place an edge at a depth the source does not have). The integrator catches the error, fills the
prerequisite, and retries.

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
how to stamp persisted knowledge based on its own loop semantics — `LEARNED` for LLM-proposed fills accepted as-is,
`CONVERSATIONAL` for human-modified proposals, etc. The graph remembers not just what it knows, but how it came to
know it.

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

The repository includes a reference `learn_gaps` tool (`examples/langchain_ollama/tools.py`) and a `DemoLearner`
(`examples/langchain_ollama/learner.py`) that exercise the full pattern: ChatOllama structured output for filling
candidates, Rich panels for human review, accept/modify/skip/reject decisions, and prerequisite handling for
reachability gaps. The langchain agent gets the `learn_gaps` tool alongside its primary tools — when a guarded
command is blocked, the agent calls `learn_gaps` to resolve the gaps and retries.

---

## Policy System

Policies live on `APPLIES_TO` relata. They define human-in-the-loop gates for specific concept relationships: which
actions require confirmation, under what cardinality conditions they fire, and what
confirmation message to surface.

### Defining Policies

```python
from vre.core.policy.models import Policy, Cardinality

Policy(
    name="confirm_file_deletion",
    requires_confirmation=True,
    trigger_cardinality=Cardinality.MULTIPLE,  # fires on recursive/glob ops
    confirmation_message="This will delete multiple files. Proceed?",
)
```

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

```python
from vre.core.policy.callback import PolicyCallback, PolicyCallContext
from vre.core.policy.models import PolicyCallbackResult


class BlockProtectedFiles:
    """Block deletion of files matching 'protected*'."""

    def __call__(self, context: PolicyCallContext) -> PolicyCallbackResult:
        command = context.tool_call.call_args[0] if context.tool_call.call_args else ""
        targets = [t for t in command.split()[1:] if not t.startswith("-")]

        for target in targets:
            if target.startswith("protected"):
                return PolicyCallbackResult(
                    passed=False,
                    message=f"'{target}' is a protected file.",
                )

        return PolicyCallbackResult(passed=True, message="No protected files affected.")
```

Callbacks are registered on a `Policy` via a dotted import path, resolved at evaluation time:

```python
Policy(
    name="protected_file_guard",
    requires_confirmation=False,  # hard block — no confirmation prompt
    trigger_cardinality=None,
    # fires on any cardinality
    callback="myproject.policies.BlockProtectedFiles",  # dotted path to the callable
    confirmation_message="Deletion blocked by protected file policy.",
)
```

A single relatum can carry multiple policies with different callbacks — one that checks file patterns, another that
checks time-of-day, another that checks user role — and each independently decides
whether its violation fires.

The repository includes a reference `protected_file_delete` callback (`examples/langchain_ollama/policies.py`) that
inspects `rm` commands across three detection modes: literal filename match, glob
expansion against the filesystem, and recursive directory inspection. It demonstrates how a callback can make nuanced,
context-aware decisions by inspecting both the command arguments and the actual
filesystem state.

### Evaluation Flow

1. **Cardinality filter** — if the policy specifies a `trigger_cardinality`, it only fires when the operation's
   cardinality matches
2. **Callback evaluation** — if a callback is registered, it runs with the full call context. `passed=True` suppresses
   the violation entirely
3. **Violation collection** — unsuppressed policies produce `PolicyViolation` objects
4. **Hard blocks vs confirmation** — violations with `requires_confirmation=False` are immediate blocks. Those with
   `requires_confirmation=True` are deferred to the `on_policy` handler

### Policy Wizard

`run_wizard(repo)` is an interactive helper for attaching policies to `APPLIES_TO` relata without manually editing seed
scripts. Construct a repository and pass it in:

```python
from vre.core.backends import SQLiteRepository
from vre.core.policy.wizard import run_wizard

with SQLiteRepository() as repo:
    run_wizard(repo)
```

It walks you through selecting source and target primitives, viewing the relata table, defining policy fields, and
persisting the result to the graph.

<img width="1968" height="1592" alt="image" src="https://github.com/user-attachments/assets/81257f0f-4273-4235-85ca-dcb50c21439b" />

---

## Integrations

The repository includes reference integrations that demonstrate how to wire VRE into real agent frameworks. These are
not part of the `vre` package — they live in the `examples/` directory and are
meant to be read, adapted, and used as starting points for your own integration.

### LangChain + Ollama Reference Agent

`examples/langchain_ollama/` contains a complete LangChain + Ollama agent that exercises all of VRE's enforcement layers
against a sandboxed filesystem.

#### Prerequisites

This example requires [Ollama](https://ollama.com/) running locally:

```bash
brew install ollama
ollama pull qwen3:8b
```

Install the example dependencies:

```bash
poetry install --extras examples
```

#### Running

```bash
# SQLite (default — no setup needed)
poetry run python -m examples.langchain_ollama.main \
    --model qwen3:8b \
    --concepts-model qwen2.5-coder:7b \
    --sandbox examples/langchain_ollama/workspace

# Neo4j
poetry run python -m examples.langchain_ollama.main \
    --backend neo4j \
    --neo4j-uri neo4j://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password \
    --model qwen3:8b \
    --concepts-model qwen2.5-coder:7b \
    --sandbox examples/langchain_ollama/workspace
```

The agent exposes a single `shell_tool` — a sandboxed subprocess executor — guarded by `vre_guard`. Every shell command
the LLM decides to run is intercepted before execution:

1. A `ConceptExtractor` sends the command to a local LLM to identify conceptual primitives (`touch foo.txt` ->
   `["create", "file"]`)
2. Those concepts are grounded against the graph
3. The epistemic trace is rendered to the terminal via `on_trace`
4. Applicable policies are evaluated
5. If a policy fires, `on_policy` prompts for confirmation before the command runs

**The agent cannot execute a command whose conceptual domain it does not understand**, and it cannot bypass policies
that require human confirmation.

#### Concept Extraction

`ConceptExtractor` (`examples/langchain_ollama/callbacks.py`) sends each command segment to a local Ollama model and
collects the conceptual primitives it identifies. The prompt includes few-shot
flag-to-concept examples (e.g. `rm -rf dir/` -> delete + directory + file) and an explicit instruction to never return
flag names as primitives.

It splits compound commands (pipes, `&&`, `;`) into segments and extracts concepts from each independently. The model is
configurable via `--concepts-model` (default `qwen2.5-coder:7b`).

`get_cardinality` is a simple rule-based function that inspects flags and globs — no LLM needed. Integrators can mix LLM
and rule-based strategies for different parameters.

#### Wiring It Together

```python
from vre.guard import vre_guard

concepts = ConceptExtractor()


@vre_guard(
    vre,
    concepts=concepts,  # LLM extracts primitives from command string
    cardinality=get_cardinality,  # inspects flags/globs -> "single" or "multiple"
    on_trace=on_trace,  # renders epistemic tree to terminal
    on_policy=on_policy,  # Rich Confirm.ask prompt
)
def shell_tool(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=sandbox)
    return result.stdout + result.stderr
```

The agent is given two tools: `shell_tool` (guarded) and `learn_gaps` (the integrator-owned learning loop).
When the guarded shell_tool blocks on knowledge gaps, the agent invokes `learn_gaps` to resolve them and retries.

### Claude Code Hook

`examples/claude-code/` contains a [PreToolUse hook](https://docs.anthropic.com/en/docs/claude-code/hooks)
for [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) that intercepts
every Bash tool call before execution and gates it through VRE grounding and policy evaluation. Unlike the LangChain
example — which uses a local Ollama model for concept extraction — this integration
lets Claude itself propose the conceptual primitives, using a two-pass protocol.

#### Install

```bash
# SQLite (default — zero config)
poetry run python examples/claude-code/claude_code.py install

# Neo4j
poetry run python examples/claude-code/claude_code.py install \
    --backend neo4j --uri neo4j://localhost:7687 --user neo4j --password password
```

This writes your backend configuration to `~/.vre/config.json` and injects a `PreToolUse` hook entry into
`~/.claude/settings.json` that matches all `Bash` tool calls. Safe to call multiple times —
existing VRE hook entries are replaced, not duplicated.

#### How It Works

The hook uses a **two-pass protocol** that lets Claude propose the concepts:

**Pass 1 — Concept Request:**

1. Claude invokes a Bash command (e.g. `rm -rf foo/`)
2. The hook sees no `# vre:` prefix and blocks (exit 2), asking Claude to identify the conceptual primitives and retry
   with a `# vre:concept1,concept2` prefix

**Pass 2 — Epistemic Check:**

3. Claude reasons about the command, identifies primitives, and retries: `# vre:delete,file,directory\nrm -rf foo/`
4. The hook extracts the concepts and grounds them against the graph
5. **If not grounded** — blocks with the full grounding trace as context
6. **If confirmation-required** — returns `permissionDecision: "ask"`, deferring to Claude Code's native TUI approval
   prompt
7. **If hard blocks or user declines** — blocks with the policy result
8. **If grounded, no violations** — allows execution with the `# vre:` prefix stripped

The `# vre:` line is a shell comment — inert if executed directly. The hook strips it via `updatedInput` before the
command runs.

<img width="1638" height="788" alt="Screenshot 2026-03-04 at 10 34 15 AM" src="https://github.com/user-attachments/assets/d8bbf86e-fe71-4fa5-b6a2-c4865aedf291" />

<img width="1627" height="780" alt="Screenshot 2026-03-04 at 10 55 10 AM" src="https://github.com/user-attachments/assets/a8c6f466-8fe2-4831-8c88-9e4d463e1f13" />

#### Uninstall

```bash
poetry run python examples/claude-code/claude_code.py uninstall
```

Removes the VRE hook entry from `~/.claude/settings.json` and leaves `~/.vre/config.json` in place.

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
│       ├── gate.py              # PolicyGate — collects violations from a trace
│       ├── callback.py          # PolicyCallContext, PolicyCallback protocol
│       └── wizard.py            # Interactive policy attachment CLI
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

examples/
├── claude-code/
│   └── claude_code.py           # Claude Code PreToolUse hook — two-pass concept protocol
└── langchain_ollama/
    ├── main.py                  # Entry point — argparse + agent setup
    ├── agent.py                 # ToolAgent — LangChain + Ollama streaming loop
    ├── tools.py                 # shell_tool (guarded) and learn_gaps (integrator loop)
    ├── callbacks.py             # ConceptExtractor, on_trace, on_policy, get_cardinality
    ├── policies.py              # Demo PolicyCallback — protected file deletion guard
    ├── learner.py               # DemoLearner — ChatOllama structured output + Rich UI
    └── repl.py                  # Streaming REPL with Rich Live display
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
