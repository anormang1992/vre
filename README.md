<img width="800" height="800" alt="vre_logo" src="https://github.com/user-attachments/assets/b5431335-c4a7-4a85-a251-601c5b11627f" />

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
    - [`on_learn`](#on_learn)
- [Auto-Learning](#auto-learning)
    - [How It Works](#how-it-works-1)
    - [Candidate Types](#candidate-types)
    - [Decisions and Provenance](#decisions-and-provenance)
    - [Two-Phase Edge Placement](#two-phase-edge-placement)
    - [The `LearningCallback` ABC](#the-learningcallback-abc)
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

After installation, download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

### Infrastructure

VRE requires a running Neo4j instance for the epistemic graph:

```bash
docker run -d \
--name neo4j \
-p 7474:7474 -p 7687:7687 \
-e NEO4J_AUTH=neo4j/password \
neo4j:latest
```

If you already have neo4j installed locally, ensure it is running and note the connection details
(URI, username, password) for use in the next steps.

### Seeding the Graph

The VRE repository includes seed scripts that populate the graph with testing scenarios. Each script clears the graph before seeding
to ensure a clean slate. See [`scripts/README.md`](scripts/README.md) for full
details.

```bash
# Fully grounded graph — 16 primitives, all at D3 with complete relata
poetry run python scripts/seed_all.py \
--neo4j-uri neo4j://localhost:7687 --neo4j-user neo4j --neo4j-password password

# Gap demonstration graph — 10 primitives, deliberately shaped to produce each gap type
poetry run python scripts/seed_gaps.py \
--neo4j-uri neo4j://localhost:7687 --neo4j-user neo4j --neo4j-password password
```

---

## Core API

### Connecting to VRE

```python
from vre import VRE
from vre.core.graph import PrimitiveRepository

repo = PrimitiveRepository(
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
    on_learn=None,    # LearningCallback — auto-learning loop for knowledge gaps
)
```

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
4. **If not grounded and `on_learn` is present** — enter the [auto-learning loop](#auto-learning)
5. **Fire `on_trace` again** — surface the post-learning epistemic result
6. **If still not grounded** — return the `GroundingResult` immediately; the function does not execute
7. **Evaluate policies** — check all `APPLIES_TO` relata for applicable policy gates
8. **If hard blocks** — return `PolicyResult(BLOCK)` immediately; `on_policy` is not consulted
9. **If confirmation required** — call `on_policy` with pending violations; block if declined or no handler
10. **If BLOCK** — return the `PolicyResult`; the function does not execute
11. **Execute** — call the original function and return its result

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


### `on_learn`

Called when grounding fails and the guard enters the auto-learning loop. See [Auto-Learning](#auto-learning) for
details.

---

## Auto-Learning

When grounding fails and an `on_learn` callback is present, VRE enters an iterative learning loop that transforms
knowledge gaps into graph growth. Rather than simply blocking the action, VRE surfaces
structured templates for each gap, invokes the callback to fill them, and persists accepted knowledge back to the
graph — then re-grounds to see if the action is now justified.

This is VRE's answer to its primary adoption bottleneck: manual graph authoring. The graph grows through use.

### How It Works

1. **Gap detected** — grounding check reveals one or more knowledge gaps
2. **Template created** — VRE generates a structured candidate template based on the gap type
3. **Callback invoked** — the integrator's `on_learn` callback receives the template, the full grounding result, and the
   specific gap. The callback fills the template (via LLM, user input, or any other
   mechanism) and returns a decision.
4. **Persistence** — accepted or modified candidates are persisted to the graph with provenance tracking
5. **Re-ground** — VRE re-checks grounding. The gap landscape may have shifted — new gaps may have appeared, existing
   ones may be resolved. The loop continues until grounded, all gaps are addressed, or
   the user rejects.

### Candidate Types

Each gap type has a corresponding candidate model. Candidates carry only what's *new* — all context (primitive IDs,
existing depths, required depths) lives on the gap itself.

| Gap Type          | Candidate               | What the Agent Fills In                                                |
|-------------------|-------------------------|------------------------------------------------------------------------|
| `ExistenceGap`    | `ExistenceCandidate`    | D1 identity for a new concept (D0 is auto-generated)                   |
| `DepthGap`        | `DepthCandidate`        | Missing depth levels with properties                                   |
| `RelationalGap`   | `RelationalCandidate`   | Missing depth levels on the edge target                                |
| `ReachabilityGap` | `ReachabilityCandidate` | Edge placement: target name, relation type, source/target depth levels |

`ExistenceCandidate`, `DepthCandidate`, and `RelationalCandidate` all use `ProposedDepth`:

```python
from vre.learning.models import ProposedDepth

ProposedDepth(
    level=DepthLevel.CAPABILITIES,
    properties={"operations": ["read", "write"], "attributes": ["size", "permissions"]},
)
```

### Decisions and Provenance

The callback returns one of four decisions, and provenance is derived from what actually happened:

| Decision   | Effect                                               | Provenance       |
|------------|------------------------------------------------------|------------------|
| `ACCEPTED` | Persist as proposed                                  | `learned`        |
| `MODIFIED` | Persist after user refinement                        | `conversational` |
| `SKIPPED`  | Intentionally dismissed — loop continues to next gap | —                |
| `REJECTED` | Discard — stops the learning loop entirely           | —                |

`SKIPPED` is particularly important for reachability gaps: the absence of an edge can itself be an enforcement
mechanism. If a concept *should not* be connected, the user skips rather than placing an
edge.

### Two-Phase Edge Placement

Reachability candidates focus solely on edge placement — they declare *where* the edge goes, not what depths need to
exist. If the source or target lacks the declared depth level, the engine
automatically synthesizes a `DepthGap` and invokes the callback to learn the missing depths *before* placing the edge.
This keeps each candidate type focused on its single concern while handling
cascading dependencies naturally.

### The `LearningCallback` ABC

```python
from vre.learning.callback import LearningCallback
from vre.learning.models import LearningCandidate, CandidateDecision


class MyLearner(LearningCallback):
    def __call__(self, candidate, grounding, gap) -> tuple[LearningCandidate | None, CandidateDecision]:
        # Fill the template, present to user, return (filled, decision)
        ...
```

`LearningCallback` is an abstract base class with `__call__` as the only required method. It also supports context
manager lifecycle via `__enter__` and `__exit__` (default no-ops) — `learn_all` wraps
the session in `with callback:`, allowing callbacks to manage state across a learning session.

### Example

The following walkthrough uses the `seed_gaps` script and attempts to create and write to a file. The learning loop
resolves several knowledge gaps through agent-user conversational turns:

1. **Existence Gap** — `write` did not exist in the graph
2. **Reachability Gap** — no edges connecting `write` and `file`
3. **Depth Gaps** — both `write` and `file` were missing the depths required by the edge placement

<img width="3372" height="906" alt="image" src="https://github.com/user-attachments/assets/2c73380d-dbf9-4acd-8f74-f492c46f468a" />

<img width="3410" height="1520" alt="image" src="https://github.com/user-attachments/assets/23a7fe9c-8ae5-490d-b7c7-7638abd0c90b" />

<img width="3406" height="850" alt="image" src="https://github.com/user-attachments/assets/8b31127f-bd50-4549-aa70-4d0bef281633" />

<img width="3404" height="1498" alt="image" src="https://github.com/user-attachments/assets/cd7852f3-df99-4a83-8070-a5293d4332c4" />

Of note: the agent correctly identified additional relata that should be attached to the `File` primitive and attempted
to record them in the `properties` object. This indicates the agent is reasoning
from within the epistemic envelope defined by the grounding trace, using neighboring primitives in the subgraph to
enrich its own proposals.

The repository includes a reference `DemoLearner` implementation (`examples/langchain_ollama/learner.py`) that uses
ChatOllama structured output to fill templates and Rich to present proposals. The
user chooses: accept, modify (provide feedback, LLM re-proposes), skip, or reject.

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

The callback receives a `PolicyCallContext` containing the tool name, the full grounding result, and the original
function arguments. It returns a `PolicyCallbackResult` — `passed=True` suppresses the
violation, `passed=False` fires it.

```python
from vre.core.policy.callback import PolicyCallback, PolicyCallContext
from vre.core.policy.models import PolicyCallbackResult


class BlockProtectedFiles:
    """Block deletion of files matching 'protected*'."""

    def __call__(self, context: PolicyCallContext) -> PolicyCallbackResult:
        command = context.call_args[0] if context.call_args else ""
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

The wizard provides an interactive CLI to attach policies to `APPLIES_TO` relata without manually editing seed scripts:

```bash
poetry run python -m vre.core.policy.wizard
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

In addition to Neo4j, this example requires [Ollama](https://ollama.com/) running locally:

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
poetry run python -m examples.langchain_ollama.main \
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
    cardinality=get_cardinality,
    # inspects flags/globs -> "single" or "multiple"
    on_trace=on_trace,  # renders epistemic tree to terminal
    on_policy=on_policy,
    # Rich Confirm.ask prompt
    on_learn=on_learn,  # auto-learning callback for knowledge gaps
)
def shell_tool(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=sandbox)
    return result.stdout + result.stderr
```

### Claude Code Hook

`examples/claude-code/` contains a [PreToolUse hook](https://docs.anthropic.com/en/docs/claude-code/hooks)
for [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) that intercepts
every Bash tool call before execution and gates it through VRE grounding and policy evaluation. Unlike the LangChain
example — which uses a local Ollama model for concept extraction — this integration
lets Claude itself propose the conceptual primitives, using a two-pass protocol.

#### Install

```bash
poetry run python examples/claude-code/claude_code.py install \
    --uri neo4j://localhost:7687 --user neo4j --password password
```

This writes your Neo4j connection details to `~/.vre/config.json` and injects a `PreToolUse` hook entry into
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

### VRE Networks

An agentic network of agents that share grounded knowledge across different epistemic graphs while applying the same
enforcement mechanisms. A concept grounded at D3 in one agent's graph carries its
epistemic justification with it — the network federates knowledge while keeping each agent's epistemic contract intact.

### Epistemic Memory

A new class of memory that stores not just information but the agent's epistemic relationship to that information.
Memories are indexed by concept and depth, decay or are reinforced based on usage and
grounding history, and affect the agent's confidence in related concepts.

---

## Tech Stack

| Concern            | Technology               |
|--------------------|--------------------------|
| Language           | Python 3.12+             |
| Epistemic graph    | Neo4j                    |
| Concept resolution | spaCy (`en_core_web_sm`) |
| Data models        | Pydantic v2              |
| Package management | Poetry                   |


---

## Project Structure

```
src/vre/
├── __init__.py                  # VRE public interface (check, learn_all, check_policy)
├── guard.py                     # vre_guard decorator (grounding → learning → policy → execution)
├── metrics.py                   # MetricsManager — best-effort grounding/learning metric updates
├── tracing.py                   # TraceWriter + TraceManager — JSONL persistence + suppression
│
├── identity/
│   ├── models.py                # AgentIdentity — stable UUID bound to a registration key
│   └── registry.py              # AgentRegistry — file-based, append-only identity persistence
│
├── core/
│   ├── models.py                # Primitive, Depth, Relatum, RelationType, DepthLevel, gaps, Provenance
│   ├── errors.py                # VREError hierarchy — typed exceptions for all failure modes
│   ├── graph.py                 # PrimitiveRepository (Neo4j)
│   ├── grounding/
│   │   ├── resolver.py          # ConceptResolver — spaCy lemmatization + name lookup
│   │   ├── engine.py            # GroundingEngine — depth-gated query, gap detection
│   │   └── models.py            # GroundingResult
│   └── policy/
│       ├── models.py            # Policy, Cardinality, PolicyResult, PolicyViolation
│       ├── gate.py              # PolicyGate — collects violations from a trace
│       ├── callback.py          # PolicyCallContext, PolicyCallback protocol
│       └── wizard.py            # Interactive policy attachment CLI
│
└── learning/
    ├── callback.py              # LearningCallback ABC
    ├── models.py                # Candidate models, CandidateDecision, LearningResult
    ├── templates.py             # template_for_gap — gap → structured candidate template
    └── engine.py                # LearningEngine — template → callback → validate → persist

scripts/
├── clear_graph.py               # Clear all primitives from the Neo4j graph
├── seed_all.py                  # Seed fully grounded graph (16 primitives)
└── seed_gaps.py                 # Seed gap-demonstration graph (10 primitives)

examples/
├── claude-code/
│   └── claude_code.py           # Claude Code PreToolUse hook — two-pass concept protocol
└── langchain_ollama/
    ├── main.py                  # Entry point — argparse + agent setup
    ├── agent.py                 # ToolAgent — LangChain + Ollama streaming loop
    ├── tools.py                 # shell_tool with vre_guard applied
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
