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

VRE addresses this directly. It imposes a contract: before an action execute the agent must demonstrate that the 
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

Depth is **monotonic**: D3 grounding implies D0–D2 are also grounded. Execution of any tool requires D3. 
An agent cannot claim to understand file deletion if it only has an identity-level model of what a file is.

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

## Installation

**Requirements:** Python 3.12+, Neo4j 5+, spaCy

```bash
# Install VRE
pip install vre

# Install the spaCy language model (required for concept resolution)
python -m spacy download en_core_web_sm

# Install demo dependencies (Rich, LangChain, Ollama)
pip install 'vre[demo]'
```

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

VRE ships with several seed scripts that populates the graph with select testing scenarios,
including a fully grounded graph and a graph with missing depth and relational requirements.

```bash
poetry run python scripts/seed_all.py --neo4j-uri <uri> --neo4j-user <user> --neo4j-password <password>
```

This creates primitives for: `operating_system`, `filesystem`, `file`, `directory`, `path`, `permission`, `user`, `group`, `create`, `read`, `write`, `delete`, `list`, `move`, `copy`.

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

### Checking Grounding Directly

```python
result = vre.check(["create", "file"])

print(result.grounded)   # True / False
print(result.resolved)   # ["create", "file"] — canonical names after resolution
print(result.gaps)       # [] or list of KnowledgeGap instances
print(result)            # Full formatted epistemic trace
```

`vre.check()` always evaluates at D3 (CONSTRAINTS). If any concept is unknown, lacks the required depth, has an unmet relational dependency, or is disconnected from the other submitted concepts, `grounded` is `False` and the corresponding gaps are surfaced.

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

# policy.action is "PASS", "PENDING", or "BLOCK"
if policy.action == "PENDING":
    print(policy.confirmation_message)
```

`cardinality` hints whether the operation targets a single entity (`"single"`) or many (`"multiple"`, e.g. recursive or glob). Policies on relata can be scoped to fire only for one cardinality or always.

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
2. **Ground at D3** — verify the full subgraph meets CONSTRAINTS depth
3. **Fire `on_trace`** — surface the epistemic result to the caller
4. **If not grounded** — return the `GroundingResult` immediately; the function does not execute
5. **Evaluate policies** — check all `APPLIES_TO` relata for applicable policy gates
6. **If PENDING** — call `on_policy` for confirmation; block if declined or no handler
7. **If BLOCK** — return the `PolicyResult`; the function does not execute
8. **Execute** — call the original function and return its result

### Parameters

```python
vre_guard(
    vre,                 # VRE instance
    concepts,            # list[str] or Callable(*args, **kwargs) -> list[str]
    cardinality=None,    # str | None or Callable(*args, **kwargs) -> str | None
    on_trace=None,       # Callable[[GroundingResult], None]
    on_policy=None,      # Callable[[str], bool]
)
```

**`concepts`** can be static or dynamic. Static is appropriate when a function always touches the same concept domain. Dynamic is appropriate when the concepts depend on the actual arguments — for example, a shell tool that must inspect the command string to know what it touches:

```python
from vre.builtins.shell import parse_bash_primitives

@vre_guard(vre, concepts=parse_bash_primitives)
def shell_tool(command: str) -> str:
    ...
```

`parse_bash_primitives` is called with the same arguments as `shell_tool`. It maps common shell executables (`rm`, `mkdir`, `cat`, etc.) to their VRE concept names.

**`cardinality`** can also be static or dynamic. When dynamic, it receives the same arguments as the decorated function:

```python
def get_cardinality(command: str) -> str:
    flags = {"-r", "-R", "-rf", "--recursive"}
    tokens = set(command.split())
    has_glob = any("*" in t for t in tokens)
    return "multiple" if (flags & tokens or has_glob) else "single"

@vre_guard(vre, concepts=parse_bash_primitives, cardinality=get_cardinality)
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
- `grounded: bool` — whether all concepts cleared D3 with no gaps
- `resolved: list[str]` — canonical primitive names (or original if unresolvable)
- `gaps: list[KnowledgeGap]` — structured gap descriptions (`ExistenceGap`, `DepthGap`, `RelationalGap`, `ReachabilityGap`)
- `trace: EpistemicResponse | None` — the full subgraph with all primitives, depths, relata, and pathway

The demo renders `on_trace` as a Rich tree showing each primitive with a dot-per-depth progress indicator and its relata:

```
VRE Epistemic Check
├── ◈ create   ● ● ● ●
│   ├── APPLIES_TO  →  file       (target D2)
│   └── REQUIRES    →  filesystem (target D3)
├── ◈ file   ● ● ● ●
│   └── CONSTRAINED_BY  →  permission  (target D3)
└── ✓ Grounded at D3 — EPISTEMIC PERMISSION GRANTED
```

Green dots (`●`) represent grounded depth levels. A red `✗` at a depth level indicates a gap. Relata flagged with `✗` indicate relational gaps where the target does not meet the required depth.

<img width="2786" height="1462" alt="image" src="https://github.com/user-attachments/assets/91d2ba34-716a-4d70-8c15-148a11e6c2b7" />

### `on_policy`

Called when a policy gate returns `PENDING` — meaning a policy on an `APPLIES_TO` relatum fired and requires human confirmation. Receives the policy's confirmation message. Returns `True` to allow execution, `False` to block.

```python
def on_policy(message: str) -> bool:
    return input(f"Policy gate: {message} [y/N]: ").strip().lower() == "y"
```

The demo uses Rich's `Confirm.ask`:

```python
from rich.prompt import Confirm

def on_policy(message: str) -> bool:
    return Confirm.ask(f"[yellow]⚠  Policy gate:[/] {message}")
```

If `on_policy` is not provided and a policy requires confirmation, the guard returns `PolicyResult(action="BLOCK", reason="Confirmation required, no handler")` and the function does not execute.

<img width="1968" height="1592" alt="image" src="https://github.com/user-attachments/assets/81257f0f-4273-4235-85ca-dcb50c21439b" />

<img width="1392" height="714" alt="image" src="https://github.com/user-attachments/assets/8b701635-d4ca-4511-98e3-cda82a5dde38" />

---

## Demo Agent

The demo ships a complete LangChain + Ollama agent that exercises all of VRE's enforcement layers against a sandboxed filesystem.

```bash
poetry run python -m demo.main \
  --neo4j-uri neo4j://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password \
  --model qwen3:8b \
  --sandbox demo/workspace
```

The agent exposes a single `shell_tool` — a sandboxed subprocess executor — guarded by `vre_guard`. Every shell command the LLM decides to run is intercepted before execution:

1. The command is parsed to extract VRE concepts (`touch foo.txt` → `["create", "file"]`)
2. Those concepts are grounded against the graph
3. The epistemic trace is rendered to the terminal via `on_trace`
4. Applicable policies are evaluated
5. If a policy fires, `on_policy` prompts for confirmation before the command runs

**The agent cannot execute a command whose conceptual domain it does not understand**, and it cannot bypass policies that require human confirmation.

### Wiring it together

```python
# demo/tools.py
from vre.guard import vre_guard

@vre_guard(
    vre,
    concepts=get_concepts,     # parses the command string → ["create", "file"]
    cardinality=get_cardinality,  # inspects flags/globs → "single" or "multiple"
    on_trace=on_trace,         # renders epistemic tree to terminal
    on_policy=on_policy,       # Rich Confirm.ask prompt
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

VRE does not require a complete or richly-detailed graph to be useful. The enforcement mechanism is structural — if a concept is not grounded at D3, the action is blocked, regardless of how much or how little else the graph contains. A minimal graph with a single primitive grounded to D3 enforces the contract correctly.

What a detailed graph adds is not stronger enforcement, but better context. More primitives, more relata, and deeper property descriptions give the agent richer epistemic material to reason from. The guard stays honest either way; a richer graph makes the agent more capable within those honest bounds.

---

## Future

### Learning through failure

When a mechanical failure occurs during execution — permission denied, missing dependency, invalid path — the failure reveals a constraint that was not modeled. The agent reasoned correctly within its knowledge boundary; the failure exposes a gap *beyond* that boundary.

A future meta-epistemic layer will treat execution failures as candidates for graph growth: the agent proposes the missing relatum (e.g. `create --[CONSTRAINED_BY]--> permission`), seeks human validation, and persists the new knowledge. Depth was honest before the failure and more complete after. No contradiction — just growth.

Provenance will travel with each piece of learned knowledge — `authored`, `learned`, or `conversational` — so the audit trail lives inside the epistemic structure itself rather than in a separate log.

### Meta-epistemic discussion mode

Structured conversation about the agent's epistemic state: asking why a concept is unknown, providing domain knowledge to populate a depth, or refining relata through dialogue. User input in this mode becomes a source for graph population rather than a trigger for action.

### Scoped policy gates for non-destructive actions

An `ActionPrimitive` subclass carrying a `required_policy_scope` field, allowing read-only or exploratory operations (list, read) to pass with lighter grounding requirements than write or delete operations, without weakening the D3 constraint for execution.

### VRE Networks

An agentic network of agents that share grounded knowledge across different epistemic graphs while applying the same enforcement mechanisms. Agents in the network expose and consume epistemic subgraphs from peer VRE instances, preserving grounding guarantees across trust boundaries. A concept grounded at D3 in one agent's graph carries its epistemic justification with it — the network does not collapse knowledge into a shared mutable store, but federates it while keeping each agent's epistemic contract intact.

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
  __init__.py              # VRE public interface
  guard.py                 # vre_guard decorator
  core/
    models.py              # Primitive, Depth, Relatum, RelationType, DepthLevel, gaps
    graph.py               # PrimitiveRepository (Neo4j)
    grounding/
      resolver.py          # ConceptResolver — spaCy lemmatization + name lookup
      engine.py            # GroundingEngine — D3 query, gap detection
      models.py            # GroundingResult
    policy/
      models.py            # Policy, Cardinality, PolicyResult
      gate.py              # PolicyGate — evaluates violations against a trace
      callback.py          # PolicyCallContext, PolicyCallback protocol
      wizard.py            # Interactive policy attachment CLI
  builtins/
    shell.py               # SHELL_ALIASES + parse_bash_primitives()

scripts/
  seed.py                  # Seed the graph with core epistemic primitives

demo/
  main.py                  # Entry point — argparse + agent setup
  agent.py                 # ToolAgent — LangChain + Ollama streaming loop
  tools.py                 # shell_tool with vre_guard applied
  callbacks.py             # on_trace (Rich tree), on_policy (Rich Confirm)
  repl.py                  # Streaming REPL with Rich Live display
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