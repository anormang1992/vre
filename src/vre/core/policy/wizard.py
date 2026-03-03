# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Policy Wizard — interactive tool for attaching policies to APPLIES_TO relata.

Run: python -m vre.core.policy.wizard
"""

from __future__ import annotations

import argparse
import importlib
from uuid import UUID

from vre.core.graph import PrimitiveRepository
from vre.core.models import DepthLevel, RelationType, Relatum
from vre.core.policy.models import Cardinality, Policy


_DEPTH_LABELS: dict[DepthLevel, str] = {
    DepthLevel.EXISTENCE:    "D0 EXISTENCE",
    DepthLevel.IDENTITY:     "D1 IDENTITY",
    DepthLevel.CAPABILITIES: "D2 CAPABILITIES",
    DepthLevel.CONSTRAINTS:  "D3 CONSTRAINTS",
    DepthLevel.IMPLICATIONS: "D4 IMPLICATIONS",
}


def _depth_label(level: DepthLevel) -> str:
    """
    Return the human-readable label for a DepthLevel, e.g. "D2 CAPABILITIES".
    """
    return _DEPTH_LABELS.get(level, f"D{int(level)}")


def _prompt(msg: str, default: str = "") -> str:
    """
    Prompt the user for input, using default if the response is blank.
    """
    if default:
        return input(f"  {msg} [{default}]: ").strip() or default
    return input(f"  {msg}: ").strip()


def _prompt_yn(msg: str, default: str = "y") -> bool:
    """
    Prompt the user for a yes/no answer; loops until a valid response is given.
    """
    options = "Y/n" if default == "y" else "y/N"
    while True:
        raw = input(f"  {msg} [{options}]: ").strip().lower() or default
        if raw in ("y", "n"):
            return raw == "y"
        print("    Enter y or n.")


def _prompt_choice(msg: str, choices: list[str], default: str) -> str:
    """
    Prompt the user to choose from a fixed set of options; loops until valid input is given.
    """
    options = "/".join(choices)
    while True:
        raw = input(f"  {msg} ({options}) [{default}]: ").strip() or default
        if raw in choices:
            return raw
        print(f"    Invalid choice. Enter one of: {options}")


def _validate_callback(path: str) -> bool:
    """
    Attempt to import and resolve a dotted-path callback string; return True if valid.
    """
    module_path, _, attr = path.rpartition(".")
    if not module_path or not attr:
        print("    Callback must be a dotted path, e.g. my_module.my_fn")
        return False
    try:
        mod = importlib.import_module(module_path)
        getattr(mod, attr)
        return True
    except ImportError as exc:
        print(f"    Import error: {exc}")
        return False
    except AttributeError as exc:
        print(f"    Attribute error: {exc}")
        return False


def _display_relata_table(
    primitive,
    name_cache: dict[UUID, str],
    repo: PrimitiveRepository,
) -> None:
    """
    Print a formatted table of all relata on the primitive, highlighting APPLIES_TO edges.
    """
    print(f"\n  Primitive: {primitive.name}  (id={primitive.id})\n")
    header = f"  {'Depth':<22} {'Relation':<16} {'Target':<22} {'Tgt Depth':<18} Policies"
    print(header)
    print("  " + "-" * 88)
    for depth in primitive.depths:
        for relatum in depth.relata:
            if relatum.target_id not in name_cache:
                found = repo.find_by_id(relatum.target_id)
                name_cache[relatum.target_id] = found.name if found else str(relatum.target_id)
            target_name = name_cache[relatum.target_id]
            marker = "  <- policies here" if relatum.relation_type == RelationType.APPLIES_TO else ""
            print(
                f"  {_depth_label(depth.level):<22}"
                f" {relatum.relation_type.value:<16}"
                f" {target_name:<22}"
                f" {_depth_label(relatum.target_depth):<18}"
                f" {len(relatum.policies)}{marker}"
            )
    print()


def _display_existing_policies(relatum: Relatum) -> None:
    """
    Print all policies currently attached to a relatum, or a notice if none exist.
    """
    if not relatum.policies:
        print("\n  (no policies currently attached)\n")
        return
    print(f"\n  Existing policies ({len(relatum.policies)}):")
    for i, p in enumerate(relatum.policies, 1):
        print(f"    {i}. {p.name}")
        print(f"       requires_confirmation: {p.requires_confirmation}")
        print(f"       trigger_cardinality:   {p.trigger_cardinality}")
        print(f"       confirmation_message:  {p.confirmation_message}")
        if p.callback:
            print(f"       callback:              {p.callback}")
    print()


def _collect_policy() -> Policy:
    """
    Interactively collect the fields for a new Policy from the user.
    """
    print("\n  --- Define new policy ---")

    # name
    while True:
        name = _prompt("Policy name").strip()
        if name:
            break
        print("    Name is required.")

    # requires_confirmation
    requires_confirmation = _prompt_yn("Requires confirmation?", default="y")

    # trigger_cardinality
    tc_str = _prompt_choice("Trigger cardinality", ["always", "single", "multiple"], "always")
    if tc_str == "single":
        trigger_cardinality: Cardinality | None = Cardinality.SINGLE
    elif tc_str == "multiple":
        trigger_cardinality = Cardinality.MULTIPLE
    else:
        trigger_cardinality = None

    # confirmation_message
    default_msg = "This action requires confirmation. Proceed?"
    confirmation_message = _prompt("Confirmation message", default_msg)

    # callback (optional, validated)
    callback: str | None = None
    cb_input = _prompt("Callback dotted path (blank to skip)").strip()
    while cb_input:
        if _validate_callback(cb_input):
            callback = cb_input
            break
        cb_input = _prompt("Callback dotted path (blank to skip)").strip()

    return Policy(
        name=name,
        requires_confirmation=requires_confirmation,
        trigger_cardinality=trigger_cardinality,
        confirmation_message=confirmation_message,
        callback=callback,
    )


def _print_policy_summary(policy: Policy) -> None:
    """
    Print a formatted summary of the given policy before asking the user to confirm.
    """
    print("\n  --- Policy summary ---")
    print(f"  name:                  {policy.name}")
    print(f"  requires_confirmation: {policy.requires_confirmation}")
    print(f"  trigger_cardinality:   {policy.trigger_cardinality}")
    print(f"  confirmation_message:  {policy.confirmation_message}")
    if policy.callback:
        print(f"  callback:              {policy.callback}")
    print()


def run_wizard(repo: PrimitiveRepository) -> None:
    """
    Run the interactive policy wizard against the given repository.
    """
    name_cache: dict[UUID, str] = {}

    print("\n=== VRE Policy Wizard ===")
    print("Attach a policy to an APPLIES_TO relatum in the graph.\n")

    # Step 2: source primitive
    while True:
        src_name = _prompt("Source primitive name").strip()
        source = repo.find_by_name(src_name)
        if source is not None:
            break
        print(f"\n  NOT FOUND: '{src_name}'")
        known = repo.list_names()
        print(f"  Known primitives: {', '.join(known)}\n")

    # Step 3: display relata table
    _display_relata_table(source, name_cache, repo)

    # Step 4: target primitive
    while True:
        tgt_name = _prompt("Target primitive name").strip()
        target = repo.find_by_name(tgt_name)
        if target is not None:
            break
        print(f"\n  NOT FOUND: '{tgt_name}'. Try again.\n")

    # Collect matching APPLIES_TO edges
    matching: list[tuple[DepthLevel, int, Relatum]] = []
    for depth in source.depths:
        for idx, relatum in enumerate(depth.relata):
            if (
                relatum.relation_type == RelationType.APPLIES_TO
                and relatum.target_id == target.id
            ):
                matching.append((depth.level, idx, relatum))

    if not matching:
        print(f"\n  No APPLIES_TO edge exists from '{source.name}' to '{target.name}'. Exiting.")
        return

    # Step 5: choose depth if ambiguous
    if len(matching) == 1:
        chosen_depth_level, _chosen_idx, chosen_relatum = matching[0]
    else:
        print("\n  Multiple APPLIES_TO edges found. Choose one:")
        for i, (lvl, _, rel) in enumerate(matching, 1):
            print(f"    {i}. {_depth_label(lvl)} -> {target.name}  (policies: {len(rel.policies)})")
        while True:
            raw = input("  Enter number: ").strip()
            try:
                sel = int(raw)
                if 1 <= sel <= len(matching):
                    chosen_depth_level, _chosen_idx, chosen_relatum = matching[sel - 1]
                    break
            except ValueError:
                pass
            print(f"    Enter a number between 1 and {len(matching)}")

    # Step 6: show existing policies
    _display_existing_policies(chosen_relatum)

    # Step 7: collect new policy
    policy = _collect_policy()

    # Step 8: confirm
    _print_policy_summary(policy)
    confirm = input("Save? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  Aborted. No changes saved.")
        return

    # Step 9: mutate in memory and persist
    chosen_relatum.policies.append(policy)
    repo.save_primitive(source)
    print(
        f"\n  Saved policy '{policy.name}' on "
        f"{source.name} --[APPLIES_TO @ {_depth_label(chosen_depth_level)}]--> {target.name}\n"
    )


def main() -> None:
    """
    Entry point: parse CLI arguments and launch the policy wizard.
    """
    parser = argparse.ArgumentParser(description="VRE Policy Wizard")
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    args = parser.parse_args()

    with PrimitiveRepository(args.neo4j_uri, args.neo4j_user, args.neo4j_password) as repo:
        run_wizard(repo)


if __name__ == "__main__":
    main()
