# VRE Trust Model

**Applies to:** VRE v1.0.x
**Status:** Normative. This document is versioned with the library and reviewed alongside every release. Where the code and this document disagree, that is a bug; report it.

---

## 1. Scope

This document states, in operational terms, what VRE guarantees, what it assumes in order to guarantee it, and what it does not guarantee at all. It is written to be checked against your own deployment rather than taken on faith. It is not a description of how VRE works internally (see the README for that) and it is not a philosophical justification of the approach. It is the contract.

The short version: VRE makes an autonomous agent's *modeled* knowledge boundary explicit, inspectable, and deterministically enforced before an action executes. It does not manufacture understanding, verify that a model is correct, or compensate for an absent or inattentive human. Everything below is an elaboration of that sentence and the line it draws.

---

## 2. The Guarantee

VRE wraps the tools an agent uses to act. Before a wrapped tool executes, VRE resolves the concepts that tool declares it touches against an authored knowledge graph, and permits execution only when those concepts are grounded to the depth the graph itself requires. When they are not, the action is blocked and the specific gap is surfaced as a structured object rather than a generic error.

Stated precisely, VRE guarantees that **a wrapped action executes only if its declared concepts are grounded, to the required depth, in the authored graph, and only if every policy attached to the traversed edges permits it.** That is the whole of it. The guarantee is about consistency with the modeled domain, enforced mechanically, and recorded with provenance. It is not a claim about the world, only about the model and the agent's conformance to it.

### Where the constraints live

The constraints VRE enforces are not a separate ruleset evaluated alongside the graph. They are carried by the graph itself. Depth requirements derive from where an edge sits, so the level to which a concept must be understood is a function of the graph's shape rather than a threshold configured beside it. Policy gates are attached to specific edges, not maintained in a list parallel to the knowledge they govern. The consequence is that the grounding check and the policy check are not two independent systems that happen to run together; they are one traversal of one structure. What an agent may do and what an agent understands are the same graph read two ways, and the permission an action receives is a property of that graph's topology, not an annotation laid over it. This is the source of the determinism described below: there is no separate policy state to fall out of sync with the model, because the policy is the model.

### Determinism

Given a fixed graph, grounding is a pure function of the submitted concepts. The same concepts checked against the same graph produce the same result every time. A result describes the graph as it exists at check time, not a property that outlives later mutation (see Section 3). There is no sampling, no threshold tuning at call time, and no model inference in the enforcement path. This is the property that distinguishes structural enforcement from instructional enforcement: a constraint expressed in the graph cannot be reasoned around, compacted out of context, or interpreted away on a given run, because it is not interpreted at all. It is resolved.

What is deterministic is the resolution, not the inputs to it. The graph is authored by a human, and the concepts checked against it are declared or extracted by the integrator; neither upstream step is deterministic, and VRE's guarantee does not reach back over either. It begins once the graph is fixed and the concepts are submitted, and it holds only insofar as that declaration is faithful (Section 3). The determinism is a property of the enforcement path, a deterministic core between two steps that are not. Read "deterministic" as a claim about how a fixed graph resolves a given query, not about how either the graph or the query came to be.

### Fail-closed behavior

Every decision point in the enforcement path defaults to refusal under uncertainty:

- An ungrounded result blocks policy evaluation entirely. Confused epistemic state never yields a permissive result.
- A policy that requires human confirmation, with no confirmation handler present, blocks.
- A confirmation handler that raises an exception blocks, rather than propagating the error past the gate.
- A policy placement that references an edge absent from the graph is rejected at construction, so a gate that would silently protect nothing fails loudly instead of failing open.
- Missing provenance on any persisted knowledge raises a typed error rather than being stored as a null.

If you disable a check, you own its absence. VRE will not pretend a disabled check passed.

### Two layers, one binding

VRE provides two things to an integrating system, and they are not equally binding. The epistemic context it can surface to a model (the structured subgraph of what is understood about the concepts in play) is **advisory**: the model may use it or ignore it. The gate is **binding**: it conditions execution and the model cannot proceed past it. Do not conflate the two. The context improves the model's chances of acting well; only the gate constrains what it can do.

---

## 3. Assumptions

The guarantee in Section 2 holds only when the following hold. Each is a real dependency, and each is outside VRE's ability to verify.

### The graph is modeled correctly

VRE enforces conformance to the authored graph. It does not assess whether the graph is right. A relationship typed as one kind when it should be another, a depth requirement set too shallow for the action it gates, or a constraint the author did not know to encode, all pass through enforcement cleanly, because enforcement has no reference point outside the graph itself. The strength of the guarantee is therefore bounded by the quality of the modeling. Grounding at the constraints level certifies that the constraints someone wrote down are present and connected, not that they are the correct or complete constraints.

### A competent human attests the knowledge

Every piece of knowledge in the graph enters through human attestation: authored directly, or proposed by a model and approved by a human at the point it is persisted. The guarantee assumes that attestation is performed by someone qualified to judge the domain. A domain expert reviewing a proposed model brings an independent understanding to compare against, and will recognize a wrong relationship as wrong. A reviewer without that understanding can only check plausibility, and fluent output is plausible by construction. The attester owns the correctness of what they approve: the type, depth, and direction of each relationship, and the properties attached to it. VRE records that attestation occurred. It does not, and cannot, record whether the attester understood what they approved.

### Declared concepts match what tools actually touch

A wrapped tool is grounded against exactly the concepts it declares. If a tool touches a concept it does not declare, VRE has no way to know, and will permit the action on an incomplete picture. The declaration is a human assertion that VRE trusts without verifying. Keeping it accurate as tools change is an integration responsibility, best enforced by tests that reconcile each tool's declared concepts against its actual interface and fail when they drift.

Of every assumption in this section, this is the one most able to defeat the guarantee silently. The gate is only ever as complete as the declaration feeding it, and the distance between what a tool declares and what it actually touches is invisible to VRE by construction. There is no runtime signal that a concept went undeclared. It is the seam where a structural guarantee leaks back into the very modeled-versus-real gap VRE exists to close, now relocated to the tool boundary. Treat the declaration as part of the enforcement surface, not as documentation about it.

### Policy callbacks are trusted integrator code

A policy callback is a callable you supply. VRE invokes it, hands it the tool call, the grounding closure, and the triggering edge, and treats its result as authoritative: a callback that fires without requiring confirmation is a hard block decided entirely by your code. VRE does not inspect, constrain, or verify what that code does. Its correctness is your responsibility, the same as any other code in your application, and it sits outside VRE's guarantee. A callback may also consult evidence VRE knows nothing about, such as runtime state, request history, or an external service, and that is a supported pattern. But whatever it consults, its result is authoritative only at the gate and only for that one evaluation. It is a point-in-time decision, not a durable property of the graph or the system.

### The knowledge store is trusted storage

VRE resolves the graph from a backend it reads but does not defend. Grounding is only as honest as the bytes it resolves. Anyone able to write to the store can lower a depth requirement, add a level that was never attested, retarget a relatum, or forge a provenance stamp, and the gate will enforce the altered shape as faithfully as an authored one. The determinism that makes a constraint impossible to reason around *at runtime* does nothing against a constraint edited out of the store *before* the run. Two things bound the exposure, and only two. First, policy is not in the store. Policies are code-resident, declared in your own imported Python and validated at construction, so a tampered graph can neither inject a callback nor re-point one; the most it can do to a gate is delete the edge the gate was placed on, and that fails loud at construction rather than silently disarming it. Second, provenance records origin but does not defend it, since a forged `AUTHORED` stamp reads identically to a real one. Everything else about the stored graph is taken on trust. Protecting the store (filesystem permissions, database access control, integrity monitoring) is a deployment responsibility outside VRE's guarantee, on the same footing as the human attestation the rest of this document rests on.

### The graph is a single mutable store, and this version does not serialize concurrent writers for you

A grounding result describes the graph at check time; it is not a property that survives later mutation. When learning persists knowledge, VRE re-validates each candidate against the *live* graph at the persistence gate, not the snapshot the gap was built from, and refuses to persist a gap the live state has already resolved. That best-effort reconciliation catches the common case of a snapshot going stale under a learning turn, but it is not full transactional isolation across the read-decide-write span, so a tight enough interleaving of writers can still race. The bundled SQLite backend uses one connection and is intended for single-process use; it rejects concurrent writers rather than queuing them. The Neo4j backend is transactional and enforces uniqueness at the store, but VRE does not coordinate writers across processes or hosts, and provides no distributed serialization in this version. Wherever more than one writer can reach a graph store, serializing those writes is the integrator's responsibility.

---

## 4. What VRE Is Not

These are the systems VRE is most often mistaken for. It is none of them, and it does not replace any of them.

**VRE is not an authorization system.** It does not decide whether a given identity may call a given tool. It decides whether an action is epistemically justified and policy-compliant with respect to the modeled domain. Identity, roles, and scopes are a separate concern that belongs to a separate layer. If you need to control *who* may act, VRE is not that control. A policy callback can, of course, consult identity and refuse on it, but authorization implemented that way is your code making your decision, authoritative only at that one gate and inheriting none of this document's guarantee. VRE supplies the hook; it does not become the authorization system by your using it as one.

**VRE is not a sandbox.** It does not isolate processes, restrict filesystem or network access, or enforce operating-system permissions. It governs whether an action is justified, not whether it is physically permitted. Mechanical containment is a separate layer that VRE assumes is present where it matters.

**VRE is not a correctness guarantee.** A grounded action is one the agent was justified in taking given the model. It can still be the wrong action, produce a wrong result, or have consequences the model did not capture. VRE narrows action to what is justified; it does not certify outcomes.

**VRE is not a hallucination filter.** It does not inspect model output for fabrication, scan generated text for accuracy, or constrain what the model says. It gates execution, not generation. A model can hallucinate freely right up to the moment it tries to act, at which point the gate engages on concepts, not on truth.

**VRE is not a substitute for human review.** This is not to say the gates are passive. A policy callback can decide on its own, inspecting tool arguments, grounded concepts, and the triggering edges, and a gate that does not require confirmation is a hard, automated block with no human in the loop. What the gates cannot do is supply human judgment where human judgment is what the decision requires. A confirmation gate surfaces the decision to a person and blocks until they consent; it does not make that judgment for them, and a person who consents without scrutiny gains nothing from having been asked. The knowledge those gates enforce is likewise attested by humans, not validated by VRE. Where a person is the intended decision-maker, the gate routes to them rather than standing in for them.

---

## 5. Failure Modes and Their Boundary

When an assumption in Section 3 is violated, the guarantee degrades in specific, bounded ways. Naming them is part of the contract.

**A mismodeled domain enforces the wrong boundary, confidently.** Because enforcement is deterministic and has no reference outside the graph, a modeling error is enforced with the same rigor as a correct model. The system will be exactly as confident about a wrong constraint as a right one. There is no internal signal that distinguishes the two; the only correction is a competent human revising the graph.

**An inattentive attester produces a graph that is trusted as if it were examined.** Provenance records that knowledge was attested, and that record reads identically whether the attester scrutinized the proposal or waved it through. The integrity of attested knowledge rests on the diligence of the attester, and that diligence is not observable in the artifact. This is a deliberate boundary, not an oversight: VRE cannot put a verifier underneath the human, because the human judgment is the floor of the system.

**An undeclared concept passes unseen.** A tool that touches more than it declares is grounded on a partial picture, and an action that should have surfaced a gap may execute without one. This failure is silent at runtime and is the reason the concept declaration deserves test coverage rather than trust.

**An unwrapped tool is ungoverned.** VRE governs the tools it wraps. A tool added to an agent without the guard is outside the model's enforcement entirely. Coverage is an integration property, not an automatic one. The guard governs the wrapper, not the underlying function's existence: `vre_guard` preserves the original through `functools.wraps`, so it stays reachable as `fn.__wrapped__`, and any caller that invokes it directly sidesteps the gate. Enforcement is cooperative by design, layered beneath mechanical and human safety rather than replacing them.

**A writable store enforces forged knowledge, faithfully.** A graph an attacker can edit is a graph VRE will resolve and enforce as though it were authored. The constraint that cannot be reasoned around at runtime can still be edited away beforehand, and provenance will not betray the edit. The integrity of the store is a deployment boundary, not a property VRE supplies. Policy is the lone exception, code-resident and never read from the store, so it can be disarmed only by deleting the edge it sits on, which fails loud at construction rather than silently at the gate.

In every one of these, the common shape is the same: VRE faithfully enforces the model it was given, and the failure lives in the gap between that model and reality, or in the human link the model depends on. VRE is honest about which side of that line it operates on. It does not claim the other side.

---

## 6. What You Get in Return

Having stated the limits plainly, here is what remains, and why it is worth the work the assumptions demand.

The cost VRE asks is front-loaded. You model the domain carefully, with a qualified person attesting the result. In exchange, every subsequent autonomous action draws on that model deterministically, without re-litigating what is grounded, and without depending on a constraint surviving in a prompt across a long context. A fixed, inspectable investment replaces a recurring, probabilistic, and fragile one. For a system that will act many times against a stable domain, that trade compounds in your favor.

Front-loaded is not the same as paid once. A domain that changes underneath the model reopens the cost, because the same determinism that enforces a correct boundary enforces a stale one with equal confidence (Section 5); a model of a moving domain is maintained, not authored once and forgotten. What the trade actually buys is a legible cost rather than a single payment. The constraint is one you revise deliberately, in one place you can audit, instead of one dispersed across prompts that erodes without notice. For a stable domain the investment is genuinely one-time. For a moving one it recurs, but it stays bounded, visible, and yours to control, which is still the trade you want.

What you hold afterward is threefold:

1. **A boundary that is explicit.** The limits of the agent's modeled understanding are an object you can read, rather than an emergent property you discover when an action goes wrong.
2. **A gate that is deterministic and fail-closed.** Enforcement does not vary with the model's reasoning quality on a given run, which is precisely the property you want when the model is one you do not fully control.
3. **An audit record honest about origin.** The question a reviewer in a regulated domain is required to answer, "why was the agent permitted to do this, and where did that knowledge come from," has a structured answer rather than a shrug.

None of this removes the human from the loop. It relocates the human's effort to where it has the most leverage, the one-time act of modeling and attestation, and makes everything downstream of that act mechanical, inspectable, and consistent. The guarantee is bounded by the human, as it should be. This document exists so that boundary is stated rather than implied.
