"""
Seed script for populating VRE with epistemic primitives.

Run: poetry run python scripts/seed.py
"""
import argparse

from vre.core.graph import PrimitiveRepository
from vre.core.models import Depth, DepthLevel, Primitive, Relatum, RelationType


def seed_operating_system(repo: PrimitiveRepository) -> Primitive:
    """
    Seed the OperatingSystem primitive at D0–D3.

    The OS is the foundational substrate — the environment that provides
    filesystems, enforces permissions, manages processes, and mediates
    all interaction between software and hardware.
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
    print(f"Saved: operating_system ({os_prim.id})")
    return os_prim


def seed_filesystem(repo: PrimitiveRepository, os_prim: Primitive) -> Primitive:
    """
    Seed the Filesystem primitive at D0–D3.

    The filesystem is the organizational substrate for persistent data.
    It depends on the OS, which provisions and manages it.
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
    print(f"Saved: filesystem ({filesystem.id})")
    return filesystem


def seed_directory(repo: PrimitiveRepository, filesystem: Primitive, path: Primitive, permission: Primitive) -> Primitive:
    """
    Seed the Directory primitive at D0–D3.

    A directory is a container within a filesystem that organizes files
    and other directories into a hierarchy.
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
    print(f"Saved: directory ({directory.id})")
    return directory


def seed_path(repo: PrimitiveRepository, filesystem: Primitive) -> Primitive:
    """
    Seed the Path primitive at D0–D3.

    A path is the addressing mechanism for locating entities within a
    filesystem's hierarchy. It is not an entity itself — it is how
    entities are named and found.
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
    print(f"Saved: path ({path.id})")
    return path


def seed_file(repo: PrimitiveRepository, filesystem: Primitive, path: Primitive, permission: Primitive) -> Primitive:
    """
    Seed the File primitive at D0–D3.

    A file is a persistent unit of data that exists within a filesystem.
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
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Can be created, read, written, deleted, moved, copied, and renamed.",
                    "operations": ["create", "read", "write", "delete", "move", "copy", "rename"],
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
                    "description": "Operations are subject to filesystem permissions, path validity, "
                                   "available disk space, and OS-level locking.",
                    "constraints": [
                        "Path must be valid for the target filesystem",
                        "Write requires write permission on the parent directory",
                        "Read requires read permission on the file",
                        "Delete requires write permission on the parent directory",
                        "File must not be locked by another process for exclusive operations",
                    ],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.CONSTRAINED_BY,
                        target_id=permission.id,
                        target_depth=DepthLevel.IDENTITY,
                        metadata={
                            "read": "requires read permission on the file",
                            "write": "requires write permission on the file",
                            "execute": "requires execute permission on the file",
                            "create": "requires write permission on the parent directory",
                            "delete": "requires write permission on the parent directory",
                            "provenance": "authored",
                        },
                    ),
                ],
            ),
        ],
    )
    repo.save_primitive(file)
    print(f"Saved: file ({file.id})")
    return file


def seed_permission(repo: PrimitiveRepository, os_prim: Primitive) -> Primitive:
    """
    Seed the Permission primitive at D0–D3.

    Permission is a general access control concept — a rule governing
    whether an actor may perform an operation on a target. The primitive
    itself is generic; domain-specific details (POSIX rwx, ACLs) belong
    on the relata that connect permission to constrained entities.
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
    print(f"Saved: permission ({permission.id})")
    return permission


def seed_create(repo: PrimitiveRepository, file: Primitive, directory: Primitive) -> Primitive:
    """
    Seed the Create primitive at D0–D3 with APPLIES_TO relata to File and Directory.
    """
    create = Primitive(
        name="create",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "An operation that brings a new entity into existence where none previously existed.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Accepts a target type, an optional destination context, and optional "
                                   "initial state. Produces a new instance of the target type.",
                    "inputs": ["target_type", "destination_context", "initial_state"],
                    "outputs": ["new_entity_instance"],
                },
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
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "The destination context must exist and permit new entities. "
                                   "The target type must be known. The entity must not already exist "
                                   "at the destination unless replacement is explicitly intended.",
                    "constraints": [
                        "Destination context must exist",
                        "Destination context must permit creation",
                        "Target type must be known",
                        "Entity must not already exist at destination unless replacement is explicit",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(create)
    print(f"Saved: create ({create.id})")
    return create


def seed_read(repo: PrimitiveRepository, file: Primitive) -> Primitive:
    """
    Seed the Read primitive at D0–D3.

    Read is the general operation of retrieving the contents or state of
    an existing entity without modifying it. Domain-specific details
    (partial reads, format interpretation) belong on the relata.
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
                            "partial": "supports offset and range for partial reads",
                            "formats": ["text", "binary"],
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "The target entity must exist. The actor must have sufficient "
                                   "permission to access it. The entity must be in a state that "
                                   "permits reading.",
                    "constraints": [
                        "Target entity must exist",
                        "Actor must have read access to the target",
                        "Target must be in a readable state",
                        "Read does not modify the entity",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(read)
    print(f"Saved: read ({read.id})")
    return read


def seed_copy(repo: PrimitiveRepository, file: Primitive, directory: Primitive, path: Primitive) -> Primitive:
    """
    Seed the Copy primitive at D0–D3.

    Copy is the general operation of duplicating an entity into a new
    context. Unlike move, the source entity is preserved.
    """
    copy = Primitive(
        name="copy",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "An operation that duplicates an entity, producing a new "
                                   "independent instance with identical contents. The source "
                                   "entity is preserved unchanged.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Accepts a source entity and a destination context. Produces "
                                   "a new entity at the destination with contents identical to "
                                   "the source. The copy is independent — subsequent changes to "
                                   "either do not affect the other.",
                    "inputs": ["source_entity", "destination_context"],
                    "outputs": ["new_entity_instance"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                    ),
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        metadata={
                            "recursive": "copying a directory copies all of its contents",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.REQUIRES,
                        target_id=path.id,
                        target_depth=DepthLevel.IDENTITY,
                        metadata={
                            "source": "path identifies the entity to copy",
                            "destination": "path identifies where to place the copy",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "The source entity must exist. The destination context must "
                                   "exist and have sufficient capacity. The actor must have read "
                                   "access to the source and write access to the destination. "
                                   "The entity must not already exist at the destination unless "
                                   "replacement is explicitly intended.",
                    "constraints": [
                        "Source entity must exist",
                        "Destination context must exist and have sufficient capacity",
                        "Actor must have read access at source and write access at destination",
                        "Entity must not exist at destination unless replacement is explicit",
                        "Copy is non-destructive to the source",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(copy)
    print(f"Saved: copy ({copy.id})")
    return copy


def seed_move(repo: PrimitiveRepository, file: Primitive, directory: Primitive, path: Primitive) -> Primitive:
    """
    Seed the Move primitive at D0–D3.

    Move is the general operation of relocating an entity from one
    context to another. The entity ceases to exist at the source
    and exists at the destination.
    """
    move = Primitive(
        name="move",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "An operation that relocates an entity from one context to "
                                   "another. The entity ceases to exist at the source location "
                                   "and exists at the destination. Content is preserved.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Accepts a target entity and a destination context. "
                                   "Produces the entity at the destination and removes it "
                                   "from the source. May also rename the entity if the "
                                   "destination implies a new identity.",
                    "inputs": ["target_entity", "destination_context"],
                    "outputs": ["relocated_entity"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                    ),
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        metadata={
                            "recursive": "moving a directory moves all of its contents",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.REQUIRES,
                        target_id=path.id,
                        target_depth=DepthLevel.IDENTITY,
                        metadata={
                            "source": "path identifies the entity to move",
                            "destination": "path identifies where to move it",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "The source entity must exist. The destination context must "
                                   "exist and permit the entity. The actor must have sufficient "
                                   "permission at both source and destination. The entity must "
                                   "not already exist at the destination unless replacement is "
                                   "explicitly intended.",
                    "constraints": [
                        "Source entity must exist",
                        "Destination context must exist and permit the entity",
                        "Actor must have access at both source and destination",
                        "Entity must not exist at destination unless replacement is explicit",
                        "Move across filesystem boundaries may degrade to copy-then-delete",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(move)
    print(f"Saved: move ({move.id})")
    return move


def seed_list(repo: PrimitiveRepository, directory: Primitive) -> Primitive:
    """
    Seed the List primitive at D0–D3.

    List is the general operation of enumerating the contents or members
    of a container. Domain-specific details belong on the relata.
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
                            "filtering": "supports glob patterns and type filtering (files only, dirs only)",
                            "metadata": "may include size, timestamps, permissions per entry",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "The container entity must exist. The actor must have "
                                   "sufficient permission to enumerate its contents. List is "
                                   "non-destructive.",
                    "constraints": [
                        "Container entity must exist",
                        "Actor must have read access to the container",
                        "List does not modify the container or its contents",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(list_prim)
    print(f"Saved: list ({list_prim.id})")
    return list_prim


def seed_delete(repo: PrimitiveRepository, file: Primitive, directory: Primitive) -> Primitive:
    """
    Seed the Delete primitive at D0–D3.

    Delete is the general operation of removing an existing entity from
    existence. It is the inverse of create.
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
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                    ),
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        metadata={
                            "recursive": "directory deletion may require recursive removal of contents",
                            "empty_check": "some systems require the directory to be empty first",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "The target entity must exist. The actor must have sufficient "
                                   "permission to remove it. Deletion is irreversible by default. "
                                   "Some entities may be protected from deletion.",
                    "constraints": [
                        "Target entity must exist",
                        "Actor must have delete access in the target's context",
                        "Deletion is irreversible by default",
                        "Some entities may be protected from deletion",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(delete)
    print(f"Saved: delete ({delete.id})")
    return delete


def seed_write(repo: PrimitiveRepository, file: Primitive) -> Primitive:
    """
    Seed the Write primitive at D0–D3.

    Write is the general operation of modifying the contents or state
    of an existing entity. Domain-specific details (append vs overwrite,
    encoding) belong on the relata.
    """
    write = Primitive(
        name="write",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "An operation that modifies the contents or state of an "
                                   "existing entity. Unlike create, the entity must already "
                                   "exist. Unlike read, the entity is changed.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Accepts a target entity and new content or state. Produces "
                                   "a modified version of the entity. Write is destructive — "
                                   "previous state may be lost unless explicitly preserved.",
                    "inputs": ["target_entity", "new_content_or_state"],
                    "outputs": ["modified_entity"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        metadata={
                            "modes": ["overwrite", "append", "insert_at_offset"],
                            "encoding": "content may require encoding (utf-8, binary)",
                            "atomicity": "write may not be atomic — partial writes are possible on failure",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "The target entity must exist. The actor must have sufficient "
                                   "permission to modify it. The entity must be in a state that "
                                   "permits modification. Previous state may be irrecoverably lost.",
                    "constraints": [
                        "Target entity must exist",
                        "Actor must have write access to the target",
                        "Target must be in a writable state",
                        "Write is destructive — previous state may be lost",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(write)
    print(f"Saved: write ({write.id})")
    return write


def seed_user(repo: PrimitiveRepository, os_prim: Primitive) -> Primitive:
    """
    Seed the User primitive at D0–D3.

    A user is the OS-recognized principal of action — the identity against
    which permissions are evaluated and ownership is attributed.
    """
    user = Primitive(
        name="user",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "An entity recognized by the operating system as a principal of "
                                   "action. Has a unique identity within the OS's user management "
                                   "system. Used to attribute ownership, enforce permissions, and "
                                   "establish the security context for running processes.",
                    "attributes": ["identity", "uid", "home_directory", "process_context"],
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Can own files and directories, belong to groups, and run "
                                   "processes. The OS evaluates permissions against the user's "
                                   "identity and group memberships when operations are attempted.",
                    "operations": ["authenticate", "own_resource", "spawn_process", "elevate"],
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
                    "description": "User identity must be recognized by the OS. Operations are "
                                   "constrained by permissions granted directly to the user or "
                                   "inherited via group membership. Superuser may bypass most "
                                   "permission constraints.",
                    "constraints": [
                        "A process executes under exactly one effective user identity at a time",
                        "User identity is resolved at process creation, not re-evaluated during execution",
                        "Privilege escalation requires an explicit OS-mediated mechanism",
                        "One user identity is designated as superuser with elevated capabilities",
                        "A user must exist in the OS identity store before processes can run under it",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(user)
    print(f"Saved: user ({user.id})")
    return user


def seed_group(repo: PrimitiveRepository, os_prim: Primitive, user: Primitive) -> Primitive:
    """
    Seed the Group primitive at D0–D3.

    A group is a named collection of users through which permissions are
    inherited collectively. The OS resolves group membership when evaluating
    access control.
    """
    group = Primitive(
        name="group",
        depths=[
            Depth(level=DepthLevel.EXISTENCE),
            Depth(
                level=DepthLevel.IDENTITY,
                properties={
                    "description": "A named collection of users recognized by the operating system. "
                                   "Groups aggregate permissions and ownership, allowing multiple users "
                                   "to share access to resources under a common identity.",
                    "attributes": ["name", "gid", "members"],
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                properties={
                    "description": "Can contain multiple users. Permissions granted to a group apply "
                                   "to all its members. The OS evaluates group membership when "
                                   "resolving access control for a user.",
                    "operations": ["add_member", "remove_member", "resolve_members"],
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=os_prim.id,
                        target_depth=DepthLevel.CAPABILITIES,
                    ),
                    Relatum(
                        relation_type=RelationType.INCLUDES,
                        target_id=user.id,
                        target_depth=DepthLevel.IDENTITY,
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                properties={
                    "description": "Group membership is the mechanism by which permissions are "
                                   "inherited collectively. Membership and identity are managed "
                                   "exclusively by the OS.",
                    "constraints": [
                        "A group identity must be unique within the OS's identity namespace",
                        "A process inherits group memberships at creation; changes require a new process context",
                        "The number of simultaneous group memberships per user is OS-bounded",
                        "A group must exist in the OS identity store before it can be referenced",
                        "Membership modification requires privileged access",
                    ],
                },
            ),
        ],
    )
    repo.save_primitive(group)
    print(f"Saved: group ({group.id})")
    return group


def main(repository: PrimitiveRepository) -> None:
    with repository as repo:
        repo.ensure_constraints()
        os_prim = seed_operating_system(repo)
        filesystem = seed_filesystem(repo, os_prim)
        path = seed_path(repo, filesystem)
        permission = seed_permission(repo, os_prim)
        directory = seed_directory(repo, filesystem, path, permission)
        file = seed_file(repo, filesystem, path, permission)
        create = seed_create(repo, file, directory)
        read = seed_read(repo, file)
        write = seed_write(repo, file)
        delete = seed_delete(repo, file, directory)
        list_prim = seed_list(repo, directory)
        move = seed_move(repo, file, directory, path)
        copy = seed_copy(repo, file, directory, path)
        user = seed_user(repo, os_prim)
        group = seed_group(repo, os_prim, user)

        # Add INCLUDES relata to filesystem now that file and directory exist.
        # Filesystem's D2 (CAPABILITIES) includes these entities at D0 (EXISTENCE).
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
        print(f"Updated: filesystem with INCLUDES relata")

        # Add APPLIES_TO relata to permission now that user, group, and action
        # primitives exist. Permission's D2 (CAPABILITIES) applies to actor types
        # at D1 (IDENTITY) and to action types at D2 (CAPABILITIES) — permission
        # governs what actions an actor is allowed to perform.
        perm_caps = next(d for d in permission.depths if d.level == DepthLevel.CAPABILITIES)
        perm_caps.relata.extend([
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=user.id,
                target_depth=DepthLevel.IDENTITY,
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=group.id,
                target_depth=DepthLevel.IDENTITY,
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=read.id,
                target_depth=DepthLevel.CAPABILITIES,
                metadata={
                    "description": "Permission governs whether an actor may perform a read operation",
                    "provenance": "authored",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=write.id,
                target_depth=DepthLevel.CAPABILITIES,
                metadata={
                    "description": "Permission governs whether an actor may perform a write operation",
                    "provenance": "authored",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=delete.id,
                target_depth=DepthLevel.CAPABILITIES,
                metadata={
                    "description": "Permission governs whether an actor may perform a delete operation",
                    "provenance": "authored",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=create.id,
                target_depth=DepthLevel.CAPABILITIES,
                metadata={
                    "description": "Permission governs whether an actor may perform a create operation",
                    "provenance": "authored",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=list_prim.id,
                target_depth=DepthLevel.CAPABILITIES,
                metadata={
                    "description": "Permission governs whether an actor may enumerate the contents of a directory",
                    "provenance": "authored",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=move.id,
                target_depth=DepthLevel.CAPABILITIES,
                metadata={
                    "description": "Permission governs whether an actor may relocate an entity across contexts",
                    "provenance": "authored",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=copy.id,
                target_depth=DepthLevel.CAPABILITIES,
                metadata={
                    "description": "Permission governs whether an actor may duplicate an entity into a new context",
                    "provenance": "authored",
                },
            ),
        ])
        repo.save_primitive(permission)
        print(f"Updated: permission with APPLIES_TO relata")

        # Add CONSTRAINED_BY relata to action primitives at D3 (CONSTRAINTS).
        # read, write, delete, and create all state permission constraints in
        # prose but have no relatum backing them — this closes that gap.
        for action_prim in (read, write, delete, create):
            action_constraints = next(d for d in action_prim.depths if d.level == DepthLevel.CONSTRAINTS)
            action_constraints.relata.append(
                Relatum(
                    relation_type=RelationType.CONSTRAINED_BY,
                    target_id=permission.id,
                    target_depth=DepthLevel.IDENTITY,
                    metadata={
                        "description": f"The {action_prim.name} action requires the actor to hold the corresponding permission on the target",
                        "provenance": "authored",
                    },
                )
            )
            repo.save_primitive(action_prim)
        print(f"Updated: read, write, delete, create with CONSTRAINED_BY permission at D3")

        list_constraints = next(d for d in list_prim.depths if d.level == DepthLevel.CONSTRAINTS)
        list_constraints.relata.append(
            Relatum(
                relation_type=RelationType.CONSTRAINED_BY,
                target_id=permission.id,
                target_depth=DepthLevel.IDENTITY,
                metadata={
                    "description": "The list action requires read permission on the target directory to enumerate its contents",
                    "provenance": "authored",
                },
            )
        )
        repo.save_primitive(list_prim)
        print(f"Updated: list with CONSTRAINED_BY permission at D3")

        move_constraints = next(d for d in move.depths if d.level == DepthLevel.CONSTRAINTS)
        move_constraints.relata.append(
            Relatum(
                relation_type=RelationType.CONSTRAINED_BY,
                target_id=permission.id,
                target_depth=DepthLevel.IDENTITY,
                metadata={
                    "description": "The move action requires write permission on both source and destination contexts — the source must permit removal and the destination must permit creation",
                    "source_permission": "write (delete-equivalent) on the source context",
                    "destination_permission": "write (create-equivalent) on the destination context",
                    "provenance": "authored",
                },
            )
        )
        repo.save_primitive(move)
        print(f"Updated: move with CONSTRAINED_BY permission at D3")

        copy_constraints = next(d for d in copy.depths if d.level == DepthLevel.CONSTRAINTS)
        copy_constraints.relata.append(
            Relatum(
                relation_type=RelationType.CONSTRAINED_BY,
                target_id=permission.id,
                target_depth=DepthLevel.IDENTITY,
                metadata={
                    "description": "The copy action requires read permission on the source and write permission on the destination context",
                    "source_permission": "read on the source entity",
                    "destination_permission": "write (create-equivalent) on the destination context",
                    "provenance": "authored",
                },
            )
        )
        repo.save_primitive(copy)
        print(f"Updated: copy with CONSTRAINED_BY permission at D3")

        print("\nDone. Seeded 16 primitives.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Grounded Graph Seeder")
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
