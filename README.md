<img width="800" height="800" alt="vre_logo" src="https://github.com/user-attachments/assets/b5431335-c4a7-4a85-a251-601c5b11627f" />

# VRE — Volute Reasoning Engine

**Epistemic enforcement for autonomous agents.**

VRE is a Python library that gives autonomous agents an explicit, inspectable model of what they know before they act. It is not a permissions system, a rules engine, or a safety classifier. It is a mechanism for making an agent's knowledge boundary a first-class object — one that can be queried, audited, and enforced at runtime.

---

## The Problem

Modern LLM-based agents fail in a specific and consistent way: they act as if they know more than they can justify.

This is not a capability problem. The models are capable. It is an *epistemic* problem — the agent has no internal representation of the boundary between what it genuinely understands and what it is confabulating. Hallucination, unsafe execution, and overconfident planning are all symptoms of the same root cause: **epistemic opacity**.

When an agent is asked to delete files, migrate a database, or execute a shell command, the question is not only "can I do this?" but "do I actually understand what I am doing well enough to do it safely?" Current systems have no mechanism to answer that second question. They proceed anyway.

This is not hypothetical. In December 2025, Amazon's Kiro agent — given operator-level access to fix a small issue in 
AWS Cost Explorer — decided the correct approach was to delete and recreate the environment entirely, causing a 
[13-hour outage](https://www.theregister.com/2026/02/20/amazon_denies_kiro_agentic_ai_behind_outage/). In February 2026, 
[OpenClaw deleted the inbox](https://techcrunch.com/2026/02/23/a-meta-ai-security-researcher-said-an-openclaw-agent-ran-amok-on-her-inbox/) of 
Summer Yue — Meta's Director of AI Alignment — after context window compaction silently discarded her instruction to 
wait for approval before taking action. The agent continued operating on a compressed history that no longer 
contained the rule. In each case, the agent acted confidently on knowledge it could not justify. The safety 
constraints were linguistic — instructions that could be forgotten, overridden, or reasoned around. 
VRE's constraints are structural.

VRE addresses this directly. It imposes a contract: before an action executes the agent must demonstrate that the 
relevant concepts are grounded in the knowledge graph at the depth required for execution. If they are not, the action 
is blocked and the gap is surfaced explicitly. The agent does not guess. It does not proceed on partial knowledge. 
It is structurally incapable of executing an action that it does not understand with respect to its epistemic model, 
and perhaps more importantly, it surfaces what it does not know. Absence of knowledge is treated as a first-class
object.

<img width="3168" height="710" alt="image" src="https://github.com/user-attachments/assets/4fedf455-a5d2-4443-acb5-ba85ac99f15c" />

---

## How It Works

### The Epistemic Graph

VRE maintains a graph of **primitives** — conceptual entities like `file`, `create`, `permission`, `directory`. These are not tools or commands. They are concepts: the things an agent reasons *about*, not the mechanisms it uses to act.

Each primitive is grounded across a hierarchy of **depth levels**:

| Depth | Name | Question answered |
|-------|---|---|
| D0    | EXISTENCE | Does this concept exist? |
| D1    | IDENTITY | What is it, in principle? |
| D2    | CAPABILITIES | What can happen to it / what can it do? |
| D3    | CONSTRAINTS | Under what conditions does that hold? |
| D4+   | IMPLICATIONS | What follows if it happens? |

Depth is **monotonic**: D3 grounding implies D0–D2 are also grounded. Depth requirements are derived from the graph 
structure itself — edges carry a source depth that determines when they become visible and a target depth that 
determines when they resolve. An integrator can also enforce a minimum depth floor (e.g. D3 for execution) as a 
secondary safety lever. An agent cannot claim to understand file deletion if it only has an identity-level model 
of what a file is.

### Relata

Primitives are connected by typed, directional, depth-aware **relata**:

```
create --[APPLIES_TO @ D2]--> file
file   --[CONSTRAINED_BY @ D3]--> permission
```

A relatum declares that understanding one concept at a given depth requires understanding another concept at a specified depth. When VRE resolves a grounding query, it follows these dependencies and checks that the entire connected subgraph meets the required depth. A relational gap — where a dependency's target is not grounded deeply enough — is surfaced as a distinct gap type.

### Policies

Policies live on `APPLIES_TO` relata. They define human-in-the-loop gates for specific concept relationships: which actions require confirmation, under what cardinality conditions they fire, and what confirmation message to surface.

```python
from vre.core.policy.models import Policy, Cardinality

Policy(
    name="confirm_file_deletion",
    requires_confirmation=True,
    trigger_cardinality=Cardinality.MULTIPLE,   # fires on recursive/glob ops
    confirmation_message="This will delete multiple files. Proceed?",
)
```

### Layered Safety

VRE is one layer of a deliberately layered safety model:

1. **Epistemic safety (VRE)** — prevents unjustified action. The agent cannot act on what it does not understand.
2. **Mechanical safety (tool constraints)** — constrains *how* the agent can act. Sandboxing, path restrictions, resource guards.
3. **Human safety (policy gates)** — requires explicit consent for elevated or destructive actions.

VRE governs only the first layer, by design. It does not replace sandboxing. It does not replace human oversight. It makes those layers more meaningful by ensuring the agent understood what it was doing when it asked for permission to act.

---

## Scope

**VRE is not a sandbox.** It does not isolate processes, restrict filesystem access, or enforce OS-level permissions. It operates at the epistemic layer — determining whether an action is justified, not whether it is physically permitted.

**VRE is not a safety classifier.** It does not scan outputs for harmful content or filter model responses. It gates execution, not generation.

**VRE is not a replacement for human oversight.** Its policy gates are a mechanism for human oversight — surfacing decisions that require consent and blocking until consent is given.

---

**Infrastructure:**

VRE requires a running Neo4j instance for the epistemic graph and (optionally) an Ollama instance for the demo agent.

```bash
# Neo4j via Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Ollama (macOS)
brew install ollama
ollama pull qwen3:8b
```

---

## Seeding the Graph

VRE ships with seed scripts that populate the graph with select testing scenarios. Each script clears the graph before seeding to ensure a clean slate. See `scripts/README.md` for full details.

```bash
# Fully grounded graph — 16 primitives, all at D3 with complete relata
python -m scripts.seed_all --neo4j-uri <uri> --neo4j-user <user> --neo4j-password <password>

# Gap demonstration graph — 10 primitives, deliberately shaped to produce each gap type
python -m scripts.seed_gaps --neo4j-uri <uri> --neo4j-user <user> --neo4j-password <password>
```

---

## Core Usage

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

An optional `agent_key` associates the VRE instance with a stable agent identity. The key is resolved via a file-based registry (`~/.vre/agents.json`) so that the same key always maps to the same UUID, even across restarts. When configured, every `GroundingResult` produced by this instance carries the agent's `agent_id`.

```python
vre = VRE(repo, agent_key="my-agent", agent_name="My Agent")

vre.identity.agent_id    # stable UUID, persisted across restarts
vre.identity.name        # "My Agent"
```

`agent_name` is a human-readable label used only on first registration — subsequent calls with the same key return the existing identity. Both parameters are optional; without `agent_key`, traces are anonymous and `vre.identity` is `None`.

You may also pass an optional `registry_path` to customize the location of the agent registry file.
By default, it is `~/.vre/agents.json`.

```python
vre = VRE(repo, agent_key="my-agent", agent_name="My Agent", registry_path="/path/to/registry.json")
```

### Checking Grounding Directly

```python
result = vre.check(["create", "file"])

print(result.grounded)   # True / False
print(result.resolved)   # ["create", "file"] — canonical names after resolution
print(result.gaps)       # [] or list of KnowledgeGap instances
print(result)            # Full formatted epistemic trace
```

`vre.check()` derives depth requirements from graph structure — edges that live at higher source depths are only 
visible when the source primitive is grounded to that depth. An optional `min_depth` parameter lets integrators enforce 
a stricter floor (e.g. D3 for execution). If any concept is unknown, lacks the required depth, has an unmet relational 
dependency, or is disconnected from the other submitted concepts, `grounded` is `False` and the corresponding gaps are surfaced.

### Using the Trace as Agent Context

`vre.check()` can be called before an agent runs to pre-load the epistemic trace into the model's context window. Rather than letting the LLM reason from general knowledge alone, you give it the graph's structured understanding of the relevant concepts — their constraints, dependencies, and relata — before it decides what to do.

```python
result = vre.check(["delete", "file"])

if result.grounded:
    # Inject the formatted trace as a system or user message
    context = str(result)   # full structured trace, formatted for readability
    response = llm.invoke([
        SystemMessage(content="You are a filesystem agent."),
        SystemMessage(content=f"Epistemic context:\n{context}"),
        HumanMessage(content=user_input),
    ])
else:
    # Surface gaps before the agent runs rather than after it tries
    for gap in result.gaps:
        print(f"Knowledge gap: {gap}")
```

This is particularly useful for planning-mode interactions: the agent receives structured knowledge of what it understands (and at what depth) before it proposes an action, rather than discovering gaps at execution time.

### Checking Policy

```python
policy = vre.check_policy(["delete", "file"], cardinality="multiple")

# policy.action is "PASS" or "BLOCK"
if policy.action == "BLOCK":
    print(policy.reason)
    for v in policy.violations:
        print(f"  - {v.message}")
```

`cardinality` hints whether the operation targets a single entity (`"single"`) or many (`"multiple"`, e.g. recursive or glob). Policies on relata can be scoped to fire only for one cardinality or always.

An optional `on_policy` callback handles violations that require human confirmation. It receives only the confirmation-required violations and returns `True` to proceed or `False` to block. Violations with `requires_confirmation=False` are hard blocks — `on_policy` is never consulted for those.

### Policy Callbacks

A `PolicyCallback` is a callable attached to a `Policy` that runs *during* evaluation to make domain-specific pass/fail decisions. This is distinct from `on_policy`, which handles human confirmation *after* violations are collected. A policy callback determines whether a violation fires at all.

The callback receives a `PolicyCallContext` containing the tool name, the full grounding result, and the original function arguments. It returns a `PolicyCallbackResult` — `passed=True` suppresses the violation, `passed=False` fires it.

#### Writing a callback

```python
from vre.core.policy.callback import PolicyCallback, PolicyCallContext
from vre.core.policy.models import PolicyCallbackResult


class BlockProtectedFiles:
    """Block deletion of files matching 'protected*'."""

    def __call__(self, context: PolicyCallContext) -> PolicyCallbackResult:
        # Extract the command from the guarded function's arguments
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

When `passed=True`, the policy is satisfied and no violation is created — the action proceeds without interruption. When `passed=False`, the violation fires and follows the normal policy flow: hard block if `requires_confirmation=False`, or deferred to `on_policy` for human confirmation if `requires_confirmation=True`.

#### Attaching a callback to a policy

Callbacks are registered on a `Policy` via a dotted import path. The path is resolved at evaluation time:

```python
from vre.core.policy.models import Policy, Cardinality

Policy(
    name="protected_file_guard",
    requires_confirmation=False,                          # hard block — no confirmation prompt
    trigger_cardinality=None,                             # fires on any cardinality
    callback="myproject.policies.BlockProtectedFiles",    # dotted path to the callable
    confirmation_message="Deletion of {action} blocked by protected file policy.",
)
```

When this policy is attached to a `delete --[APPLIES_TO]--> file` relatum in the graph, every delete operation targeting files will invoke `BlockProtectedFiles`. If the callback returns `passed=False`, the action is blocked immediately (since `requires_confirmation=False`). If the callback returns `passed=True`, no violation fires and the action proceeds.

#### Evaluation flow

Policy callbacks participate in a layered evaluation:

1. **Cardinality filter** — if the policy specifies a `trigger_cardinality`, it only fires when the operation's cardinality matches
2. **Callback evaluation** — if a callback is registered, it runs with the full call context. `passed=True` suppresses the violation entirely
3. **Violation collection** — unsuppressed policies produce `PolicyViolation` objects
4. **Hard blocks vs confirmation** — violations with `requires_confirmation=False` are immediate blocks. Those with `requires_confirmation=True` are deferred to the `on_policy` handler

This means a single relatum can carry multiple policies with different callbacks — one that checks file patterns, another that checks time-of-day, another that checks user role — and each independently decides whether its violation fires.

#### Demo example

The demo ships with a `protected_file_delete` callback (`examples/langchain_ollama/policies.py`) that inspects `rm` commands across three detection modes: literal filename match, glob expansion against the filesystem, and recursive directory inspection. It demonstrates how a callback can make nuanced, context-aware decisions by inspecting both the command arguments and the actual filesystem state:

```python
# Registered on the delete → file APPLIES_TO relatum via seed_all.py
Policy(
    name="protected_file_delete",
    requires_confirmation=False,
    trigger_cardinality=None,
    callback="examples.langchain_ollama.policies.protected_file_delete",
    confirmation_message="Deletion of {action} blocked — protected files at risk.",
)
```

When an agent runs `rm *.txt` in a directory containing `protected_config.txt`, the callback expands the glob, detects the protected file, and returns `passed=False` — blocking the deletion before it reaches the shell.

---

## The `vre_guard` Decorator

`vre_guard` is the primary integration point. It wraps any callable and gates it behind a grounding check
and a policy evaluation before the function body executes. This is designed to wrap the tools your agent uses to 
act on the world, ensuring that every action is epistemically justified and compliant with your defined policies.

```python
from vre.guard import vre_guard

@vre_guard(vre, concepts=["write", "file"])
def write_file(path: str, content: str) -> str:
    ...
```

Each call runs the following sequence:

1. **Resolve concepts** — map names to canonical primitives via the graph
2. **Ground** — verify the subgraph meets depth requirements (graph-derived + optional `min_depth` floor)
3. **Fire `on_trace`** — surface the epistemic result to the caller
4. **If not grounded and `on_learn` is present** — enter the auto-learning loop (see [Auto-Learning](#auto-learning))
5. **Fire `on_trace` again** — surface the post-learning epistemic result
6. **If still not grounded** — return the `GroundingResult` immediately; the function does not execute
7. **Evaluate policies** — check all `APPLIES_TO` relata for applicable policy gates
8. **If hard blocks** — return `PolicyResult(BLOCK)` immediately; `on_policy` is not consulted
9. **If confirmation required** — call `on_policy` with pending violations; block if declined or no handler
10. **If BLOCK** — return the `PolicyResult`; the function does not execute
11. **Execute** — call the original function and return its result

### Parameters

```python
vre_guard(
    vre,                 # VRE instance
    concepts,            # list[str] or Callable(*args, **kwargs) -> list[str]
    cardinality=None,    # str | None or Callable(*args, **kwargs) -> str | None
    min_depth=None,      # DepthLevel | None — enforces a minimum depth floor
    on_trace=None,       # Callable[[GroundingResult], None]
    on_policy=None,      # Callable[[list[PolicyViolation]], bool]
    on_learn=None,       # LearningCallback — auto-learning loop for knowledge gaps
)
```

**`concepts`** can be static or dynamic. Static is appropriate when a function always touches the same concept domain. Dynamic is appropriate when the concepts depend on the actual arguments — for example, a shell tool that must inspect the command string to know what it touches. Any callable that accepts the same arguments as the decorated function and returns `list[str]` works:

```python
concepts = ConceptExtractor()   # LLM-based — see examples/langchain_ollama/callbacks.py

@vre_guard(vre, concepts=concepts)
def shell_tool(command: str) -> str:
    ...
```

VRE does not own concept extraction. The integrator decides how to map tool arguments to primitives — an LLM call, a static alias table, a rule engine, or any combination. VRE only requires the result: a list of concept names to ground.

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

---

## Callbacks

### `on_trace`

Called after grounding, whether grounded or not. Receives the full `GroundingResult`. Use this to render the epistemic trace to your UI.

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
- `gaps: list[KnowledgeGap]` — structured gap descriptions (`ExistenceGap`, `DepthGap`, `RelationalGap`, `ReachabilityGap`)
- `trace: EpistemicResponse | None` — the full subgraph with all primitives, depths, relata, and pathway
- `agent_id: UUID | None` — the stable agent identifier, when the VRE instance was created with an `agent_key`

The demo renders `on_trace` as a Rich tree showing each primitive with a dot-per-depth progress indicator and its relata:

```
VRE Epistemic Check
├── ◈ create   ● ● ● ●
│   ├── APPLIES_TO  →  file       (target D2)
│   └── REQUIRES    →  filesystem (target D3)
├── ◈ file   ● ● ● ●
│   └── CONSTRAINED_BY  →  permission  (target D3)
└── ✓ Grounded — EPISTEMIC PERMISSION GRANTED
```

Green dots (`●`) represent grounded depth levels. A red `✗` at a depth level indicates a gap. Relata flagged with `✗` indicate relational gaps where the target does not meet the required depth.

<img width="2786" height="1462" alt="image" src="https://github.com/user-attachments/assets/91d2ba34-716a-4d70-8c15-148a11e6c2b7" />

### `on_policy`

Called when policy evaluation produces violations that require human confirmation (`requires_confirmation=True`). Receives only the confirmation-required violations — hard blocks (`requires_confirmation=False`) are handled before `on_policy` is ever consulted. Returns `True` to proceed, `False` to block.

```python
from vre.core.policy.models import PolicyViolation

def on_policy(violations: list[PolicyViolation]) -> bool:
    for v in violations:
        answer = input(f"Policy gate: {v.message} [y/N]: ").strip().lower()
        if answer != "y":
            return False
    return True
```

The demo uses Rich's `Confirm.ask`:

```python
def on_policy(violations: list[PolicyViolation]) -> bool:
    for v in violations:
        if not Confirm.ask(f"[yellow]⚠  Policy gate:[/] {v.message}"):
            return False
    return True
```

If `on_policy` is not provided and a policy requires confirmation, the guard returns `PolicyResult(action=PolicyAction.BLOCK, reason="Confirmation required, no handler")` and the function does not execute.

<img width="1968" height="1592" alt="image" src="https://github.com/user-attachments/assets/81257f0f-4273-4235-85ca-dcb50c21439b" />

<img width="1392" height="714" alt="image" src="https://github.com/user-attachments/assets/8b701635-d4ca-4511-98e3-cda82a5dde38" />

### `on_learn`

Called when grounding fails and the guard enters the auto-learning loop. See [Auto-Learning](#auto-learning) for details.

---

## Auto-Learning

When grounding fails and an `on_learn` callback is present, VRE enters an iterative learning loop that transforms knowledge gaps into graph growth. Rather than simply blocking the action, VRE surfaces structured templates for each gap, invokes the callback to fill them, and persists accepted knowledge back to the graph — then re-grounds to see if the action is now justified.

This is VRE's answer to its primary adoption bottleneck: manual graph authoring. The graph grows through use.

### How it works

1. **Gap detected** — grounding check reveals one or more knowledge gaps
2. **Template created** — VRE generates a structured candidate template based on the gap type
3. **Callback invoked** — the integrator's `on_learn` callback receives the template, the full grounding result, and the specific gap. The callback fills the template (via LLM, user input, or any other mechanism) and returns a decision.
4. **Persistence** — accepted or modified candidates are persisted to the graph with provenance tracking
5. **Re-ground** — VRE re-checks grounding. The gap landscape may have shifted — new gaps may have appeared, existing ones may be resolved. The loop continues until grounded, all gaps are addressed, or the user rejects.

### Candidate types

Each gap type has a corresponding candidate model. Candidates carry only what's *new* — all context (primitive IDs, existing depths, required depths) lives on the gap itself.

| Gap Type | Candidate | What the agent fills in |
|---|---|---|
| `ExistenceGap` | `ExistenceCandidate` | D1 identity for a new concept (D0 is auto-generated) |
| `DepthGap` | `DepthCandidate` | Missing depth levels with properties |
| `RelationalGap` | `RelationalCandidate` | Missing depth levels on the edge target |
| `ReachabilityGap` | `ReachabilityCandidate` | Edge placement: target name, relation type, source/target depth levels |

`ExistenceCandidate`, `DepthCandidate`, and `RelationalCandidate` all use `ProposedDepth` — the agent-facing depth model:

```python
from vre.learning.models import ProposedDepth

ProposedDepth(
    level=DepthLevel.CAPABILITIES,
    properties={"operations": ["read", "write"], "attributes": ["size", "permissions"]},
)
```

`ProposedDepth` carries only `level` and `properties` — descriptive attributes intrinsic to the concept at that depth. Relata and provenance are structural concerns handled by the engine during persistence.

### Decisions and provenance

The callback returns one of four decisions, and provenance is derived from what actually happened:

| Decision | Effect | Provenance |
|---|---|---|
| `ACCEPTED` | Persist as proposed | `learned` |
| `MODIFIED` | Persist after user refinement | `conversational` |
| `SKIPPED` | Intentionally dismissed — loop continues to next gap | — |
| `REJECTED` | Discard — stops the learning loop entirely | — |

`SKIPPED` is particularly important for reachability gaps: the absence of an edge can itself be an enforcement mechanism. If a concept *should not* be connected, the user skips rather than placing an edge.

### Two-phase edge placement

Reachability candidates focus solely on edge placement — they declare *where* the edge goes, not what depths need to exist. If the source or target lacks the declared depth level, the engine automatically synthesizes a `DepthGap` and invokes the callback to learn the missing depths *before* placing the edge. If depth learning is rejected, the edge placement is abandoned.

This keeps each candidate type focused on its single concern while handling cascading dependencies naturally.

### The `LearningCallback` ABC

```python
from vre.learning.callback import LearningCallback
from vre.learning.models import LearningCandidate, CandidateDecision

class MyLearner(LearningCallback):
    def __call__(self, candidate, grounding, gap) -> tuple[LearningCandidate | None, CandidateDecision]:
        # Fill the template, present to user, return (filled, decision)
        ...
```

`LearningCallback` is an abstract base class with `__call__` as the only required method. It also supports context manager lifecycle via `__enter__` and `__exit__` (with default no-op implementations) — `learn_all` wraps the session in `with callback:`, allowing callbacks to manage state across a learning session without the guard knowing about callback internals.

### Example

The following example uses the `seed_gaps` script and attempts to create and write to a file. The learning loop flows through several knowledge gaps and agent-user conversational turns to resolve the gaps: 
1. Existence Gap (write did not exist in the graph)
2. Reachability Gap (no edges connecting write and file)
3. Depth Gap(s) (Both write and file were missing the depths required by the edge placement)

<img width="3372" height="906" alt="image" src="https://github.com/user-attachments/assets/2c73380d-dbf9-4acd-8f74-f492c46f468a" />

<img width="3410" height="1520" alt="image" src="https://github.com/user-attachments/assets/23a7fe9c-8ae5-490d-b7c7-7638abd0c90b" />

<img width="3406" height="850" alt="image" src="https://github.com/user-attachments/assets/8b31127f-bd50-4549-aa70-4d0bef281633" />

<img width="3404" height="1498" alt="image" src="https://github.com/user-attachments/assets/cd7852f3-df99-4a83-8070-a5293d4332c4" />

Of special note, is that the agent correctly identified additional relata that should be attached to the File primitive; however, because the current iteration only defines `name` and `properties` for the `ProposedDepth`, the agent tried to record the relata in the properties object. This indicates that the agent is indeed reasoning from within the epistemic envelope defined by the grounding trace and is using neighboring primitives in the subgraph to try and enrich its own proposals. An issue has been captured to formalize this behavior in the future by expanding the `ProposedDepth` model schema to include `proposed_relata`.

### Demo implementation

The demo's `DemoLearner` uses ChatOllama structured output to fill templates and Rich to present proposals:

1. The user is prompted to enter learning mode (once per session, via `__enter__`/`__exit__` lifecycle)
2. The LLM fills the candidate template using the epistemic trace as context
3. The proposal is rendered as a Rich panel showing all proposed knowledge
4. The user chooses: accept, modify (provide feedback → LLM re-proposes), skip, or reject
5. Accepted knowledge is persisted and grounding is re-checked

---

## Demo Agent

The demo ships a complete LangChain + Ollama agent that exercises all of VRE's enforcement layers against a sandboxed filesystem.

```bash
poetry run python -m examples.langchain_ollama.main \
  --neo4j-uri neo4j://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password \
  --model qwen3:8b \
  --concepts-model qwen2.5-coder:7b \
  --sandbox examples/langchain_ollama/workspace
```

The agent exposes a single `shell_tool` — a sandboxed subprocess executor — guarded by `vre_guard`. Every shell command the LLM decides to run is intercepted before execution:

1. A `ConceptExtractor` sends the command to a local LLM to identify conceptual primitives (`touch foo.txt` → `["create", "file"]`)
2. Those concepts are grounded against the graph
3. The epistemic trace is rendered to the terminal via `on_trace`
4. Applicable policies are evaluated
5. If a policy fires, `on_policy` prompts for confirmation before the command runs

**The agent cannot execute a command whose conceptual domain it does not understand**, and it cannot bypass policies that require human confirmation.

### Concept extraction

The demo uses `ConceptExtractor` (`examples/langchain_ollama/callbacks.py`) — a callable class that sends each command segment to a local Ollama model and collects the conceptual primitives it identifies. The prompt includes few-shot flag-to-concept examples (e.g. `rm -rf dir/` → delete + directory + file) and an explicit instruction to never return flag names as primitives.

`ConceptExtractor` is constructed once at startup and reused across calls. It splits compound commands (pipes, `&&`, `;`) into segments and extracts concepts from each independently. The model is configurable via `--concepts-model` (default `qwen2.5-coder:7b`).

`get_cardinality` is a simple rule-based function that inspects flags and globs — no LLM needed. Integrators can mix LLM and rule-based strategies for different parameters.

### Wiring it together

```python
# examples/langchain_ollama/tools.py
from vre.guard import vre_guard

concepts = ConceptExtractor()  # LLM-based concept extraction (Ollama)

@vre_guard(
    vre,
    concepts=concepts,            # LLM extracts primitives from command string
    cardinality=get_cardinality,  # inspects flags/globs → "single" or "multiple"
    on_trace=on_trace,            # renders epistemic tree to terminal
    on_policy=on_policy,          # Rich Confirm.ask prompt
    on_learn=on_learn,            # auto-learning callback for knowledge gaps
)
def shell_tool(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=sandbox)
    return result.stdout + result.stderr
```

---

## Policy Wizard

Policies are attached to `APPLIES_TO` relata in the graph. The wizard provides an interactive path to add policies without manually editing the seed script:

```bash
poetry run python -m vre.core.policy.wizard
```

The wizard walks you through:
1. Select a source primitive (e.g. `delete`)
2. View its relata table with depth labels and current policy counts
3. Select a target primitive (e.g. `file`)
4. Define policy fields interactively — name, cardinality, confirmation message, optional callback
5. Confirm and persist to the graph

The result is a policy that fires on the `delete --[APPLIES_TO]--> file` edge. The next time an agent attempts a delete 
operation, the guard evaluates this policy and, if it applies, surfaces the confirmation prompt before execution.

---

## Claude Code Integration

VRE ships with a [PreToolUse hook](https://docs.anthropic.com/en/docs/claude-code/hooks) for [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) that intercepts every Bash tool call before execution and gates it through VRE grounding and policy evaluation. Unlike the demo agent — which uses a local Ollama model for concept extraction — the Claude Code integration lets Claude itself propose the conceptual primitives, using a two-pass protocol.

### Install the hook

```bash
python examples/claude-code/claude_code.py install \
  --uri neo4j://localhost:7687 --user neo4j --password password
```

This does two things:

1. Writes your Neo4j connection details to `~/.vre/config.json`
2. Injects a `PreToolUse` hook entry into `~/.claude/settings.json` that matches all `Bash` tool calls

The hook command uses the absolute path of the current Python interpreter, so it runs in the same virtualenv where VRE is installed. Safe to call multiple times — existing VRE hook entries are replaced, not duplicated.

### How the hook works

The hook uses a **two-pass protocol** that lets Claude — the LLM — propose the concepts rather than relying on a static command-to-concept map:

**Pass 1 — concept request:**

1. Claude invokes a Bash command (e.g. `rm -rf foo/`)
2. The hook sees no `# vre:` prefix and blocks (exit 2), writing an instruction to stderr asking Claude to identify the conceptual primitives and retry with a `# vre:concept1,concept2` prefix

**Pass 2 — epistemic check:**

3. Claude reasons about the command, identifies primitives, and retries: `# vre:delete,file,directory\nrm -rf foo/`
4. The hook extracts the concepts from the prefix and grounds them against the graph
5. **If not grounded** — the hook exits with code 2 and writes the full grounding trace to stderr, which Claude Code feeds back to the model as context. The command does not execute.
6. **If confirmation-required violations exist** — the hook returns `permissionDecision: "ask"`, deferring to Claude Code's native TUI approval prompt. All confirmation messages are listed so the user can make an informed decision.
7. **If hard blocks or user declines** — the hook exits with code 2 and the policy result is fed to the model.
8. **If grounded with no policy violations** — the hook exits with code 0, `permissionDecision: "allow"`, and `updatedInput` with the `# vre:` prefix stripped. The executed command is clean.

The `# vre:` line is a shell comment — inert if executed directly. The hook uses Claude Code's `updatedInput` mechanism to strip it before the command runs, so the actual executed command never contains the prefix.

The hook fails open when the VRE config file is absent. Empty commands are allowed through without a concept request.

<img width="1638" height="788" alt="Screenshot 2026-03-04 at 10 34 15 AM" src="https://github.com/user-attachments/assets/d8bbf86e-fe71-4fa5-b6a2-c4865aedf291" />

<img width="1627" height="780" alt="Screenshot 2026-03-04 at 10 55 10 AM" src="https://github.com/user-attachments/assets/a8c6f466-8fe2-4831-8c88-9e4d463e1f13" />


### Remove the hook

```bash
python examples/claude-code/claude_code.py uninstall
```

This removes the VRE hook entry from `~/.claude/settings.json` and leaves `~/.vre/config.json` in place.

### Notes
- The two-pass protocol adds one round-trip per novel command. Once Claude has seen the pattern, it tends to include the `# vre:` prefix proactively.
- The current hook is designed for Bash commands. It can be extended to other tool types by adding additional hook entries with different matchers.

---

## Knowledge Gaps

When a grounding check fails, VRE returns structured gap objects rather than a generic error. There are four gap types:

| Type | Meaning |
|---|---|
| `ExistenceGap` | The concept is not in the graph at all |
| `DepthGap` | The concept exists but is not grounded to the required depth |
| `RelationalGap` | A relatum's target does not meet the depth required by that edge |
| `ReachabilityGap` | The concept is not connected to the other submitted concepts |

Gaps are not failures to be hidden. They are information. An existence gap on `network` tells you the agent has no 
epistemic model of networking — not that the request was malformed. The agent can surface this gap to the user, initiate a learning flow, or escalate to a human. The gap is the signal.

### The gate holds at any graph depth

VRE does not require a complete or richly-detailed graph to be useful. The enforcement mechanism is structural — depth requirements are derived from the graph itself (edge placement gates visibility) and optionally raised by the integrator via `min_depth`. A minimal graph with a single primitive and its edges enforces the contract correctly.

What a detailed graph adds is not stronger enforcement, but better context. More primitives, more relata, and deeper property descriptions give the agent richer epistemic material to reason from. The guard stays honest either way; a richer graph makes the agent more capable within those honest bounds.

---

## Future

### Learning through failure

When a mechanical failure occurs during execution — permission denied, missing dependency, invalid path — the failure reveals a constraint that was not modeled. The agent reasoned correctly within its knowledge boundary; the failure exposes a gap *beyond* that boundary.

This is distinct from the [auto-learning loop](#auto-learning), which addresses gaps discovered *before* execution. Learning through failure would be triggered *after* execution — the graph said "go ahead" but reality disagreed. The agent proposes the missing relatum (e.g. `create --[CONSTRAINED_BY]--> permission`), seeks human validation, and persists the new knowledge. Depth was honest before the failure and more complete after. No contradiction — just growth.

### VRE Networks

An agentic network of agents that share grounded knowledge across different epistemic graphs while applying the same enforcement mechanisms. Agents in the network expose and consume epistemic subgraphs from peer VRE instances, preserving grounding guarantees across trust boundaries. A concept grounded at D3 in one agent's graph carries its epistemic justification with it — the network does not collapse knowledge into a shared mutable store, but federates it while keeping each agent's epistemic contract intact.

### Epistemic Memory

A new class of memory that stores not just information but the agent's epistemic relationship to that information.
Memories are indexed by concept and depth, and can be queried by the agent to inform its reasoning process. 
Memories will decay or be reinforced based on usage and grounding history and affect the agent's confidence in 
related concepts.

---

## Tech Stack

| Concern | Technology |
|---|---|
| Language | Python 3.12+ |
| Epistemic graph | Neo4j |
| Concept resolution | spaCy (`en_core_web_sm`) |
| Data models | Pydantic v2 |
| Agent framework (demo) | LangChain + Ollama |
| Demo UI | Rich |
| Package management | Poetry |

---

## Project Structure

```
src/vre/
  __init__.py              # VRE public interface (check, learn, learn_all, check_policy)
  guard.py                 # vre_guard decorator (grounding → learning → policy → execution)
  identity/
    models.py              # AgentIdentity — stable UUID bound to a registration key
    registry.py            # AgentRegistry — file-based, append-only identity persistence
  core/
    models.py              # Primitive, Depth, Relatum, RelationType, DepthLevel, gaps, Provenance
    graph.py               # PrimitiveRepository (Neo4j)
    grounding/
      resolver.py          # ConceptResolver — spaCy lemmatization + name lookup
      engine.py            # GroundingEngine — depth-gated query, gap detection
      models.py            # GroundingResult
    policy/
      models.py            # Policy, Cardinality, PolicyResult, PolicyViolation, PolicyCallbackResult
      gate.py              # PolicyGate — collects violations from a trace
      callback.py          # PolicyCallContext, PolicyCallback protocol
      wizard.py            # Interactive policy attachment CLI
  learning/
    callback.py            # LearningCallback ABC — __call__, __enter__, __exit__
    models.py              # Candidate models, CandidateDecision, LearningResult
    templates.py           # TemplateFactory — gap → structured candidate template
    engine.py              # LearningEngine — template → callback → validate → persist

scripts/
  clear_graph.py           # Clear all primitives from the Neo4j graph
  seed_all.py              # Seed fully grounded graph (16 primitives)
  seed_gaps.py             # Seed gap-demonstration graph (10 primitives)

examples/
  claude-code/
    claude_code.py         # Claude Code PreToolUse hook — two-pass concept protocol
  langchain_ollama/
    main.py                # Entry point — argparse + agent setup
    agent.py               # ToolAgent — LangChain + Ollama streaming loop
    tools.py               # shell_tool with vre_guard applied
    callbacks.py           # ConceptExtractor, on_trace, on_policy, get_cardinality, make_on_learn
    policies.py            # Demo PolicyCallback — protected file deletion guard
    learner.py             # DemoLearner — ChatOllama structured output + Rich UI
    repl.py                # Streaming REPL with Rich Live display
```

---

## Guiding Principle

> **The agent must never act as if it knows more than it can justify.**

VRE exists to enforce that rule — not as a policy, but as a structural property of the system.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your proposed changes. For major changes, 
please discuss them in an issue first to ensure alignment with the project's goals and architecture. 

Aside from the expected bug fixes and optimizations, here are some areas where contributions would be 
particularly valuable:

- Additional seed scripts for more complex domains (e.g. networking, databases, cloud infrastructure)
- Integration examples with other python agent frameworks (e.g. AutoGPT, BabyAGI) or tool libraries
  - Any integration submissions should also include a demo that exercises the integration in a meaningful way and 
  demonstrates that the epistemic resolution behavior works as intended.
- VRE integration into other language environments (Node.js, Go, etc.)

This is a project that I am passionate about and is the culmination of almost 10 years of philosophical thought. 
I hope to connect with other like-minded community members who prioritize safety and epistemic integrity in 
autonomous agentic systems. 

I look forward to seeing how this evolves!
