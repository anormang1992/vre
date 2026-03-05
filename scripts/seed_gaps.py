"""
Seed script for demonstrating depth-gated traversal and gap detection.

Seeds a deliberately shaped graph (10 primitives) where each action
primitive is truncated or structured to produce a specific gap type
when queried by the grounding engine.

See scripts/README.md for the full primitive table and expected gap scenarios.

Run: poetry run python scripts/seed_gaps.py
"""
import argparse

from scripts.clear_graph import clear_graph
from vre.core.graph import PrimitiveRepository
from vre.core.models import Depth, DepthLevel, Primitive, Relatum, RelationType


# ── Fully grounded substrates (D0–D3) ─────────────────────────────────────


def seed_operating_system(repo: PrimitiveRepository) -> Primitive:
    """
    Seed the OperatingSystem primitive at D0–D3.

    Fully grounded substrate. Provides the environment that provisions
    filesystems, enforces permissions, and manages processes.
    """
    os_prim = Primitive(
        name="operating_system",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "The software layer that manages hardware resources and provides "
                                   "services to applications. Mediates all access to storage, memory, "
                                   "network, and peripherals.",
                    "attributes": ["kernel", "userspace", "processes", "users", "permissions_model"],
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Provides filesystems, process management, user/group identity, "
                                   "permission enforcement, environment variables, and inter-process "
                                   "communication.",
                    "services": [
                        "filesystem_provision",
                        "process_management",
                        "permission_enforcement",
                        "user_identity",
                        "resource_allocation",
                    ],
                },
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "Behavior varies across OS families (POSIX vs Windows). Resource "
                                   "limits, security policies, and kernel capabilities bound what is "
                                   "possible. Superuser/root can bypass most permission constraints.",
                    "constraints": [
                        "Behavior is OS-family dependent (POSIX, Windows, etc.)",
                        "Kernel enforces resource limits (open files, memory, CPU)",
                        "Security policies (SELinux, AppArmor) may further restrict operations",
                        "Superuser can bypass most permission constraints",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(os_prim)
    print(f"  operating_system  D0–D3  (substrate)")
    return os_prim


def seed_filesystem(repo: PrimitiveRepository, os_prim: Primitive) -> Primitive:
    """
    Seed the Filesystem primitive at D0–D3.

    Fully grounded substrate. Depends on the OS at D2.
    """
    filesystem = Primitive(
        name="filesystem",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "A hierarchical system for organizing, storing, and retrieving "
                                   "persistent data. Provides the namespace (paths) and structure "
                                   "(directories) through which files are addressed.",
                    "attributes": ["root", "hierarchy", "mount_point", "type"],
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Organizes data into files and directories via a path-based "
                                   "hierarchy. Supports mounting, traversal, and metadata tracking "
                                   "(timestamps, ownership, permissions).",
                    "operations": [
                        "mount",
                        "traverse",
                        "resolve_path",
                        "track_metadata",
                    ],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=os_prim.id,
                        target_depth=DepthLevel.CAPABILITIES,
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "Filesystem type determines supported features (max path length, "
                                   "case sensitivity, symlinks, permissions model). Must be mounted "
                                   "before access. Storage capacity is finite.",
                    "constraints": [
                        "Filesystem must be mounted before access",
                        "Storage capacity is finite",
                        "Max path length and filename length are filesystem-dependent",
                        "Case sensitivity is filesystem-dependent",
                        "Permissions model varies by filesystem type",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(filesystem)
    print(f"  filesystem        D0–D3  (substrate)")
    return filesystem


def seed_path(repo: PrimitiveRepository, filesystem: Primitive) -> Primitive:
    """
    Seed the Path primitive at D0–D3.

    Fully grounded structural. Depends on filesystem at D2.
    """
    path = Primitive(
        name="path",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "A string that uniquely addresses a location within a "
                                   "filesystem's hierarchy. Composed of segments separated "
                                   "by a delimiter. Can be absolute (from root) or relative "
                                   "(from a working context).",
                    "attributes": ["segments", "delimiter", "absolute", "relative"],
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Can be resolved to a filesystem entity, joined with other "
                                   "paths, normalized, and decomposed into parent and basename. "
                                   "Supports extension extraction and glob pattern matching.",
                    "operations": ["resolve", "join", "normalize", "decompose", "match"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=filesystem.id,
                        target_depth=DepthLevel.IDENTITY,
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "Maximum length and valid characters are filesystem-dependent. "
                                   "Delimiter is OS-dependent (/ vs \\). Resolution may fail if "
                                   "the target does not exist. Symlinks may cause a path to "
                                   "resolve to a different location than its literal segments imply.",
                    "constraints": [
                        "Maximum length is filesystem-dependent",
                        "Valid characters are filesystem-dependent",
                        "Delimiter is OS-dependent (/ vs \\)",
                        "Resolution may fail if the target does not exist",
                        "Symlinks may cause non-literal resolution",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(path)
    print(f"  path              D0–D3  (structural)")
    return path


def seed_permission(repo: PrimitiveRepository, os_prim: Primitive) -> Primitive:
    """
    Seed the Permission primitive at D0–D3.

    Fully grounded structural. Depends on OS at D2.
    """
    permission = Primitive(
        name="permission",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "A rule that governs whether an actor is allowed to perform "
                                   "a specific operation on a specific target. Expressed as a "
                                   "relationship between actor, operation, and target with a "
                                   "grant or deny outcome.",
                    "attributes": ["actor", "operation", "target", "grant_or_deny"],
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Can be checked, granted, revoked, and inherited. Multiple "
                                   "permission models exist; the specific model is determined "
                                   "by the system that enforces the permission.",
                    "operations": ["check", "grant", "revoke", "inherit"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=os_prim.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        metadata={
                            "enforcement": "kernel-level",
                            "models": ["posix_rwx", "acl", "capabilities"],
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "Only the granting authority can modify permissions. "
                                   "Inherited permissions may be overridden by explicit grants. "
                                   "Absence of an explicit rule may imply denial or allowance "
                                   "depending on the model. Some operations may require elevated "
                                   "authority regardless of existing grants.",
                    "constraints": [
                        "Only the granting authority can modify permissions",
                        "Inherited permissions may be overridden by explicit grants",
                        "Default-deny vs default-allow depends on the model",
                        "Some operations require elevated authority regardless of existing grants",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(permission)
    print(f"  permission        D0–D3  (structural)")
    return permission


# ── Fully grounded entity (D0–D3) ─────────────────────────────────────────


def seed_directory(
    repo: PrimitiveRepository,
    filesystem: Primitive,
    path: Primitive,
    permission: Primitive,
) -> Primitive:
    """
    Seed the Directory primitive at D0–D3.

    Fully grounded entity. Target for safe operations (list) and
    destructive operations (delete, create). Deep enough (D3) to
    satisfy any target_depth requirement from APPLIES_TO edges.
    """
    directory = Primitive(
        name="directory",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "A named container within a filesystem that holds files and "
                                   "other directories, forming the hierarchical structure of "
                                   "the filesystem's namespace.",
                    "attributes": ["path", "name", "parent", "children"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.REQUIRES,
                        target_id=path.id,
                        target_depth=DepthLevel.IDENTITY,
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Can be created, deleted, listed, renamed, and traversed. "
                                   "Serves as the scope and destination context for file operations.",
                    "operations": ["create", "delete", "list", "rename", "traverse"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=filesystem.id,
                        target_depth=DepthLevel.IDENTITY,
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "Must have a valid path within a mounted filesystem. Deleting "
                                   "requires the directory to be empty or deletion to be recursive. "
                                   "The root directory cannot be deleted.",
                    "constraints": [
                        "Path must be valid within the filesystem",
                        "Parent directory must exist for creation",
                        "Delete requires directory to be empty or explicitly recursive",
                        "Root directory cannot be deleted",
                        "Listing requires read permission on the directory",
                        "Creation requires write permission on the parent directory",
                    ],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.CONSTRAINED_BY,
                        target_id=permission.id,
                        target_depth=DepthLevel.IDENTITY,
                        metadata={
                            "list": "requires read permission on the directory",
                            "create_child": "requires write permission on the directory",
                            "delete": "requires write permission on the parent directory",
                            "rename": "requires write permission on both source and destination parent",
                            "provenance": "authored",
                        },
                    ),
                ],
            ),
        ],
    )
    repo.save_primitive(directory)
    print(f"  directory         D0–D3  (entity, fully grounded)")
    return directory


# ── Shallow entity (D0–D1) ────────────────────────────────────────────────


def seed_file(repo: PrimitiveRepository, path: Primitive) -> Primitive:
    """
    Seed the File primitive at D0–D1 only.

    Deliberately shallow. When an action's APPLIES_TO targets file at
    target_depth D2, file's contiguous max (D1) is insufficient →
    RelationalGap.
    """
    file = Primitive(
        name="file",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "A persistent, named unit of data stored on a filesystem and addressed by path.",
                    "attributes": ["path", "name", "extension", "size", "content"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.REQUIRES,
                        target_id=path.id,
                        target_depth=DepthLevel.IDENTITY,
                    ),
                ],
            ),
        ],
    )
    repo.save_primitive(file)
    print(f"  file              D0–D1  (shallow → RelationalGap when targeted at D2)")
    return file


# ── Safe actions (D0–D2) ──────────────────────────────────────────────────


def seed_read(repo: PrimitiveRepository, file: Primitive) -> Primitive:
    """
    Seed the Read primitive at D0–D2.

    Safe action. APPLIES_TO lives at D2 (source_depth=D2), so the edge
    is visible when read is grounded to D2. Targets file at target_depth
    D2, but file is only at D1 → RelationalGap(read→file, req=D2, cur=D1).
    """
    read = Primitive(
        name="read",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "An operation that retrieves the contents or state of an "
                                   "existing entity without modifying it.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Accepts a target entity and returns its contents or state. "
                                   "Read is inherently non-destructive — the target is unchanged "
                                   "after the operation.",
                    "inputs": ["target_entity"],
                    "outputs": ["entity_contents_or_state"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        metadata={
                            "retrieval": "returns file contents as bytes or decoded text",
                            "provenance": "authored",
                        },
                    ),
                ],
            ),
        ],
    )
    repo.save_primitive(read)
    print(f"  read              D0–D2  (safe action, APPLIES_TO@D2 → file)")
    return read


def seed_list(repo: PrimitiveRepository, directory: Primitive) -> Primitive:
    """
    Seed the List primitive at D0–D2.

    Safe action. APPLIES_TO lives at D2 (source_depth=D2), targeting
    directory at target_depth D2. Directory is grounded to D3, so
    D3 >= D2 → clean pass, no gaps.
    """
    list_prim = Primitive(
        name="list",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "An operation that enumerates the contents or members of a "
                                   "container entity. Returns a collection of contained entities "
                                   "without modifying the container or its contents.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Accepts a container entity and returns its contents. May "
                                   "support filtering, sorting, and recursive enumeration. "
                                   "List is non-destructive.",
                    "inputs": ["container_entity"],
                    "outputs": ["collection_of_contained_entities"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        metadata={
                            "contents": "returns files and subdirectories",
                            "recursive": "may enumerate contents of subdirectories recursively",
                            "provenance": "authored",
                        },
                    ),
                ],
            ),
        ],
    )
    repo.save_primitive(list_prim)
    print(f"  list              D0–D2  (safe action, APPLIES_TO@D2 → directory)")
    return list_prim


# ── Destructive actions ───────────────────────────────────────────────────


def seed_delete(repo: PrimitiveRepository, file: Primitive, directory: Primitive) -> Primitive:
    """
    Seed the Delete primitive at D0–D2 only.

    Destructive action. In the fully grounded graph (seed_all), delete's
    APPLIES_TO edges live at D3 (CONSTRAINTS). Here D3 is omitted, so
    those edges simply don't exist — delete has no visible connection to
    its targets → ReachabilityGap on file or directory.
    """
    delete = Primitive(
        name="delete",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "An operation that removes an existing entity from existence. "
                                   "The inverse of create. After deletion, the entity no longer "
                                   "exists in its context.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Accepts a target entity and removes it. May support "
                                   "recursive deletion for composite entities. Deletion is "
                                   "irreversible unless the system provides recovery mechanisms.",
                    "inputs": ["target_entity"],
                    "outputs": ["confirmation_of_removal"],
                },
                # No relata — APPLIES_TO lives at D3 in seed_all, omitted here.
            ),
        ],
    )
    repo.save_primitive(delete)
    print(f"  delete            D0–D2  (destructive, APPLIES_TO@D3 omitted → ReachabilityGap)")
    return delete


def seed_create(repo: PrimitiveRepository, file: Primitive, directory: Primitive) -> Primitive:
    """
    Seed the Create primitive at D0–D1 + D3 (non-contiguous).

    Destructive action. Has identity (D0–D1) but is missing D2
    (CAPABILITIES). APPLIES_TO lives at D3 (source_depth=D3), but
    contiguous max is D1 (D2 absent breaks the chain). The engine
    gates the D3 edges → DepthGap(create, req=D3, cur=D1).

    Since the APPLIES_TO edges are gated, file and directory are
    unreachable from create → ReachabilityGap on the target.
    """
    create = Primitive(
        name="create",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "An operation that brings a new entity into existence where "
                                   "none previously existed.",
                },
            ),
            # D2 (CAPABILITIES) deliberately omitted — breaks contiguity.
            # The agent knows create exists and what it is, but hasn't
            # learned its capabilities. This gates all D3 edges.
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={},
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        metadata={
                            "default_destination": "current working directory when no path is specified",
                            "provenance": "authored",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        metadata={
                            "default_destination": "current working directory when no path is specified",
                            "provenance": "authored",
                        },
                    ),
                ],
            ),
        ],
    )
    repo.save_primitive(create)
    print(f"  create            D0–D1+D3  (destructive, APPLIES_TO@D3 gated — D2 missing)")
    return create


# ── Main ──────────────────────────────────────────────────────────────────


def main(repository: PrimitiveRepository) -> None:
    with repository as repo:
        repo.ensure_constraints()
        deleted = clear_graph(repo)

        print(f"Cleared {deleted} existing primitive(s).")
        print("Seeding gap-demonstration graph (10 primitives):\n")

        # Substrates
        os_prim = seed_operating_system(repo)
        filesystem = seed_filesystem(repo, os_prim)
        path = seed_path(repo, filesystem)
        permission = seed_permission(repo, os_prim)

        # Entities
        directory = seed_directory(repo, filesystem, path, permission)
        file = seed_file(repo, path)

        # Actions
        read = seed_read(repo, file)
        list_prim = seed_list(repo, directory)
        delete = seed_delete(repo, file, directory)
        create = seed_create(repo, file, directory)

        # Add INCLUDES relata to filesystem now that file and directory exist.
        fs_caps = next(d for d in filesystem.depths if d.level == DepthLevel.CAPABILITIES)
        fs_caps.relata.extend([
            Relatum(
                relation_type=RelationType.INCLUDES,
                target_id=file.id,
                target_depth=DepthLevel.EXISTENCE,
            ),
            Relatum(
                relation_type=RelationType.INCLUDES,
                target_id=directory.id,
                target_depth=DepthLevel.EXISTENCE,
            ),
        ])
        repo.save_primitive(filesystem)
        print(f"\n  Updated: filesystem with INCLUDES → file, directory")

        print("""
Done. Seeded 10 primitives.

Expected gap scenarios:
  ["list", "directory"]    → clean pass
  ["read", "file"]         → RelationalGap (file too shallow)
  ["delete", "file"]       → ReachabilityGap (APPLIES_TO@D3 omitted)
  ["delete", "directory"]  → ReachabilityGap (APPLIES_TO@D3 omitted)
  ["create", "file"]       → DepthGap + ReachabilityGap (edge gated)
  ["create", "directory"]  → DepthGap + ReachabilityGap (edge gated)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth-Gated Gap Demonstration Seeder")
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
