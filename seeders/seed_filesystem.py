"""
Seeder for the filesystem domain.

Upserts a fully grounded epistemic graph for operating system filesystem
concepts (OS, filesystem, path, permission, ownership, directory, file,
symlink, user, group, process) and their actions (create, read, write,
delete, list, move, copy, modify, execute). Idempotent — re-running
preserves existing primitive ids by name. Does not clear the graph;
compose with `scripts/clear_graph.py` if a clean slate is wanted.

Run: python seeders/seed_filesystem.py
"""
import argparse

from vre.core.backends import Neo4jRepository, Repository
from vre.core.models import Depth, DepthLevel, Primitive, Provenance, ProvenanceSource, Relatum, RelationType

SEED_PROVENANCE = Provenance(source=ProvenanceSource.AUTHORED)


def seed_operating_system(repo: Repository) -> Primitive:
    """
    Seed the OperatingSystem primitive at D0–D3.

    The OS is the foundational substrate — the environment that provides
    filesystems, enforces permissions, manages processes, and mediates
    all interaction between software and hardware.
    """
    os_prim = Primitive(
        name="operating_system",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The software layer that manages hardware resources and provides "
                                   "services to applications. Mediates all access to storage, memory, "
                                   "network, and peripherals.",
                    "kernel": "The core component that mediates hardware access and enforces privilege boundaries",
                    "userspace": "The layer where applications run, separated from kernel by privilege boundary",
                    "processes": "Independently scheduled units of execution managed by the kernel",
                    "users": "Named identities against which permissions are evaluated",
                    "permissions_model": "The mechanism by which the OS governs access to resources",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Provides filesystems, process management, user/group identity, "
                                   "permission enforcement, environment variables, and inter-process "
                                   "communication.",
                    "filesystem_provision": "Creates and manages filesystems for persistent data storage",
                    "process_management": "Creates, schedules, and terminates processes",
                    "permission_enforcement": "Evaluates and enforces access control on every operation",
                    "user_identity": "Maintains the identity store for users and groups",
                    "resource_allocation": "Distributes CPU, memory, and I/O bandwidth among processes",
                },
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Behavior varies across OS families (POSIX vs Windows). Resource "
                                   "limits, security policies, and kernel capabilities bound what is "
                                   "possible. Superuser/root can bypass most permission constraints.",
                    "os_family_dependent": "Behavior is OS-family dependent (POSIX, Windows, etc.)",
                    "resource_limits": "Kernel enforces resource limits (open files, memory, CPU)",
                    "security_policies": "Security policies (SELinux, AppArmor) may further restrict operations",
                    "superuser_bypass": "Superuser can bypass most permission constraints",
                },
            ),
        ],
    )
    os_prim = repo.upsert_primitive(os_prim)
    print(f"Saved: operating_system ({os_prim.id})")
    return os_prim


def seed_filesystem(repo: Repository, os_prim: Primitive) -> Primitive:
    """
    Seed the Filesystem primitive at D0–D3.

    The filesystem is the organizational substrate for persistent data.
    It depends on the OS, which provisions and manages it.
    """
    filesystem = Primitive(
        name="filesystem",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "A hierarchical system for organizing, storing, and retrieving "
                                   "persistent data. Provides the namespace (paths) and structure "
                                   "(directories) through which files are addressed.",
                    "root": "The top-level entry point of the hierarchy, from which all paths descend",
                    "hierarchy": "A tree structure of directories containing files and other directories",
                    "mount_point": "The location in the OS namespace where the filesystem is attached",
                    "type": "The filesystem format (ext4, APFS, NTFS) determining features and limits",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Organizes data into files and directories via a path-based "
                                   "hierarchy. Supports mounting, traversal, and metadata tracking "
                                   "(timestamps, ownership, permissions).",
                    "mount": "Attach the filesystem to a location in the OS directory hierarchy",
                    "traverse": "Navigate the path-based hierarchy to locate entities",
                    "resolve_path": "Translate a path string into a reference to a specific entity",
                    "track_metadata": "Maintain timestamps, ownership, and permissions on entities",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=os_prim.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "management": "OS provisions, mounts, and manages filesystem lifecycle",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Filesystem type determines supported features (max path length, "
                                   "case sensitivity, symlinks, permissions model). Must be mounted "
                                   "before access. Storage capacity is finite.",
                    "must_be_mounted": "Filesystem must be mounted before access",
                    "finite_capacity": "Storage capacity is finite",
                    "path_length_limit": "Max path length and filename length are filesystem-dependent",
                    "case_sensitivity": "Case sensitivity is filesystem-dependent",
                    "permissions_model": "Permissions model varies by filesystem type",
                },
            ),
        ],
    )
    filesystem = repo.upsert_primitive(filesystem)
    print(f"Saved: filesystem ({filesystem.id})")
    return filesystem


def seed_path(repo: Repository, filesystem: Primitive) -> Primitive:
    """
    Seed the Path primitive at D0–D3.

    A path is the addressing mechanism for locating entities within a
    filesystem's hierarchy. It is not an entity itself — it is how
    entities are named and found.
    """
    path = Primitive(
        name="path",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "A string that uniquely addresses a location within a "
                                   "filesystem's hierarchy. Composed of segments separated "
                                   "by a delimiter. Can be absolute (from root) or relative "
                                   "(from a working context).",
                    "segments": "Ordered components of the path separated by the delimiter",
                    "delimiter": "The character separating path segments (/ on POSIX, \\ on Windows)",
                    "absolute": "A path rooted at the filesystem root, unambiguous regardless of context",
                    "relative": "A path interpreted relative to a working directory or context",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Can be resolved to a filesystem entity, joined with other "
                                   "paths, normalized, and decomposed into parent and basename. "
                                   "Supports extension extraction and glob pattern matching.",
                    "resolve": "Translate the path into a reference to the filesystem entity it names",
                    "join": "Concatenate path segments to form a longer path",
                    "normalize": "Collapse redundant separators and resolve . and .. segments",
                    "decompose": "Split into parent directory path and basename",
                    "match": "Test whether the path matches a glob or regex pattern",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=filesystem.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "description": "Path semantics (delimiter, max length, case sensitivity) "
                                           "are determined by the filesystem",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Maximum length and valid characters are filesystem-dependent. "
                                   "Delimiter is OS-dependent (/ vs \\). Resolution may fail if "
                                   "the target does not exist. Symlinks may cause a path to "
                                   "resolve to a different location than its literal segments imply.",
                    "max_length": "Maximum length is filesystem-dependent",
                    "valid_characters": "Valid characters are filesystem-dependent",
                    "delimiter_os_dependent": "Delimiter is OS-dependent (/ vs \\)",
                    "resolution_failure": "Resolution may fail if the target does not exist",
                    "symlink_indirection": "Symlinks may cause non-literal resolution",
                },
            ),
        ],
    )
    path = repo.upsert_primitive(path)
    print(f"Saved: path ({path.id})")
    return path


def seed_permission(repo: Repository, os_prim: Primitive) -> Primitive:
    """
    Seed the Permission primitive at D0–D3.

    Permission is a general access control concept — a rule governing
    whether an actor may perform an operation on a target. The primitive
    itself is generic; domain-specific details (POSIX rwx, ACLs) belong
    on the relata that connect permission to constrained entities.
    """
    permission = Primitive(
        name="permission",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "A rule that governs whether an actor is allowed to perform "
                                   "a specific operation on a specific target. Expressed as a "
                                   "relationship between actor, operation, and target with a "
                                   "grant or deny outcome.",
                    "actor": "The user or group identity attempting the operation",
                    "operation": "The specific action being attempted (read, write, execute, etc.)",
                    "target": "The entity the operation is being performed on",
                    "grant_or_deny": "The binary outcome of evaluating the permission rule",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Can be checked, granted, revoked, and inherited. Multiple "
                                   "permission models exist; the specific model is determined "
                                   "by the system that enforces the permission.",
                    "check": "Evaluate whether an actor is permitted to perform an operation on a target",
                    "grant": "Allow an actor to perform an operation on a target",
                    "revoke": "Remove a previously granted permission",
                    "inherit": "Derive permissions from a parent context or group membership",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=os_prim.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "enforcement": "kernel-level",
                            "models": ["posix_rwx", "acl", "capabilities"],
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Only the granting authority can modify permissions. "
                                   "Inherited permissions may be overridden by explicit grants. "
                                   "Absence of an explicit rule may imply denial or allowance "
                                   "depending on the model. Some operations may require elevated "
                                   "authority regardless of existing grants.",
                    "granting_authority": "Only the granting authority can modify permissions",
                    "inheritance_override": "Inherited permissions may be overridden by explicit grants",
                    "default_policy": "Default-deny vs default-allow depends on the model",
                    "elevated_authority": "Some operations require elevated authority regardless of existing grants",
                },
            ),
        ],
    )
    permission = repo.upsert_primitive(permission)
    print(f"Saved: permission ({permission.id})")
    return permission


def seed_ownership(repo: Repository, os_prim: Primitive) -> Primitive:
    """
    Seed the Ownership primitive at D0–D3.

    Ownership is the association between an actor and a filesystem entity
    that determines the default authority relationship. It is the primary
    input to permission evaluation: the POSIX model checks owner first,
    then group, then other.
    """
    ownership = Primitive(
        name="ownership",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The association between an actor (user or group) and a "
                                   "filesystem entity that determines the default authority "
                                   "relationship. Ownership is the primary input to permission "
                                   "evaluation: the POSIX model evaluates user-owner permissions "
                                   "first, then group-owner, then other.",
                    "owner_user": "The user identity designated as the entity's owner",
                    "owner_group": "The group identity designated as the entity's group-owner",
                    "entity": "The filesystem entity that is owned",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Can be queried, transferred (chown), and used as the basis "
                                   "for permission evaluation. The owner has special privileges: "
                                   "only the owner (or superuser) can change permissions on an entity.",
                    "query": "Determine the current owner user and group of an entity",
                    "transfer": "Change the owner user or group of an entity (chown/chgrp)",
                    "evaluate": "Resolve which permission class (owner/group/other) applies to an actor",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=os_prim.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "enforcement": "Ownership is tracked and enforced by the OS/filesystem",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Every filesystem entity has exactly one owning user and one "
                                   "owning group. Only the owner or superuser can transfer ownership. "
                                   "Ownership is set at creation time and inherited from the creating "
                                   "process's effective identity.",
                    "single_owner": "Every filesystem entity has exactly one owning user and one owning group",
                    "transfer_authority": "Only the current owner or superuser can transfer ownership",
                    "os_restrictions": "Ownership transfer may be further restricted by OS policy",
                    "creation_inheritance": "Ownership is set at creation time from the creating process's effective identity",
                    "implicit_authority": "The owner has implicit authority to modify the entity's permissions",
                },
            ),
        ],
    )
    ownership = repo.upsert_primitive(ownership)
    print(f"Saved: ownership ({ownership.id})")
    return ownership


def seed_directory(repo: Repository, filesystem: Primitive, path: Primitive, permission: Primitive) -> Primitive:
    """
    Seed the Directory primitive at D0–D3.

    A directory is a container within a filesystem that organizes files
    and other directories into a hierarchy.
    """
    directory = Primitive(
        name="directory",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "A named container within a filesystem that holds files and "
                                   "other directories, forming the hierarchical structure of "
                                   "the filesystem's namespace.",
                    "path": "The directory's location within the filesystem hierarchy",
                    "name": "The final segment of the directory's path",
                    "parent": "The directory that contains this directory (except for root)",
                    "children": "The set of files and subdirectories contained within",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.REQUIRES,
                        target_id=path.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Can be created, deleted, listed, renamed, and traversed. "
                                   "Serves as the scope and destination context for file operations.",
                    "create": "Bring a new directory into existence at a specified path",
                    "delete": "Remove the directory and optionally its contents",
                    "list": "Enumerate the files and subdirectories contained within",
                    "rename": "Change the directory's name within its parent context",
                    "traverse": "Navigate into and through the directory hierarchy",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=filesystem.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "description": "Directory existence and behavior depend on the hosting filesystem",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Must have a valid path within a mounted filesystem. Deleting "
                                   "requires the directory to be empty or deletion to be recursive. "
                                   "The root directory cannot be deleted.",
                    "path_validity": "Path must be valid within the filesystem",
                    "parent_exists": "Parent directory must exist for creation",
                    "delete_empty_or_recursive": "Delete requires directory to be empty or explicitly recursive",
                    "root_protected": "Root directory cannot be deleted",
                    "list_read_permission": "Listing requires read permission on the directory",
                    "create_write_permission": "Creation requires write permission on the parent directory",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.CONSTRAINED_BY,
                        target_id=permission.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "list": "requires read permission on the directory",
                            "create_child": "requires write permission on the directory",
                            "delete": "requires write permission on the parent directory",
                            "rename": "requires write permission on both source and destination parent",
                        },
                    ),
                ],
            ),
        ],
    )
    directory = repo.upsert_primitive(directory)
    print(f"Saved: directory ({directory.id})")
    return directory


def seed_file(repo: Repository, filesystem: Primitive, path: Primitive, permission: Primitive) -> Primitive:
    """
    Seed the File primitive at D0–D3.

    A file is a persistent unit of data that exists within a filesystem.
    """
    file = Primitive(
        name="file",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "A persistent, named unit of data stored on a filesystem and addressed by path.",
                    "path": "Absolute or relative location within the filesystem hierarchy",
                    "name": "The final segment of the path, including any extension",
                    "extension": "Suffix indicating file type or format",
                    "size": "Amount of data stored, in bytes",
                    "content": "The actual data stored in the file",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.REQUIRES,
                        target_id=path.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Can be created, read, written, deleted, moved, copied, and renamed.",
                    "create": "Bring a new file into existence with optional initial content",
                    "read": "Retrieve the file's contents without modifying them",
                    "write": "Modify the file's contents (overwrite, append, or insert)",
                    "delete": "Remove the file from the filesystem",
                    "move": "Relocate the file to a different path",
                    "copy": "Duplicate the file to a new path, preserving the original",
                    "rename": "Change the file's name within its parent directory",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=filesystem.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "description": "File existence, naming, and behavior depend on the hosting filesystem",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Operations are subject to filesystem permissions, path validity, "
                                   "available disk space, and OS-level locking.",
                    "path_validity": "Path must be valid for the target filesystem",
                    "write_permission": "Write requires write permission on the file or parent directory",
                    "read_permission": "Read requires read permission on the file",
                    "delete_permission": "Delete requires write permission on the parent directory",
                    "locking": "File must not be locked by another process for exclusive operations",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.CONSTRAINED_BY,
                        target_id=permission.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "read": "requires read permission on the file",
                            "write": "requires write permission on the file",
                            "execute": "requires execute permission on the file",
                            "create": "requires write permission on the parent directory",
                            "delete": "requires write permission on the parent directory",
                        },
                    ),
                ],
            ),
        ],
    )
    file = repo.upsert_primitive(file)
    print(f"Saved: file ({file.id})")
    return file


def seed_symlink(repo: Repository, filesystem: Primitive, path: Primitive, permission: Primitive) -> Primitive:
    """
    Seed the Symlink primitive at D0–D3.

    A symlink is a filesystem entity that acts as an indirect reference
    to another entity via a target path. Its identity is defined by where
    it points, not what it contains.
    """
    symlink = Primitive(
        name="symlink",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "A filesystem entity that acts as an indirect reference to "
                                   "another entity via a target path. A symlink's identity is "
                                   "defined by where it points, not what it contains. It introduces "
                                   "a layer of indirection between a path and its resolved target.",
                    "path": "The symlink's own location within the filesystem hierarchy",
                    "name": "The final segment of the symlink's path",
                    "target_path": "The path the symlink references, which may or may not exist",
                    "is_dangling": "Whether the target path resolves to an existing entity",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.REQUIRES,
                        target_id=path.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "usage": "Symlink requires both its own path and a target path",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Can be created, deleted, read (to obtain the target path), "
                                   "and dereferenced (to follow to the target entity). Symlinks "
                                   "are transparent to most operations — reading through a symlink "
                                   "reads the target.",
                    "create": "Create a new symlink pointing to a specified target path",
                    "delete": "Remove the symlink without affecting the target entity",
                    "readlink": "Retrieve the target path stored in the symlink",
                    "dereference": "Follow the symlink to its target entity for transparent access",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=filesystem.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "description": "Symlink existence and resolution depend on the hosting filesystem",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The target path need not exist at creation time, producing a "
                                   "dangling symlink. Circular chains of symlinks cause resolution "
                                   "failure. Maximum symlink depth is OS-enforced. Permissions on "
                                   "the symlink itself are typically ignored; the target's "
                                   "permissions govern access.",
                    "dangling_valid": "Target path need not exist (dangling symlinks are valid)",
                    "circular_resolution": "Circular symlink chains cause resolution failure (ELOOP)",
                    "max_depth": "Maximum symlink resolution depth is OS-enforced",
                    "target_permissions": "Symlink permissions are typically ignored; target permissions govern access",
                    "delete_independence": "Deleting a symlink does not affect the target entity",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.CONSTRAINED_BY,
                        target_id=permission.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "readlink": "read permission on the symlink",
                            "create": "write permission on the parent directory",
                            "delete": "write permission on the parent directory",
                            "dereference": "permissions evaluated on the resolved target, not the symlink",
                        },
                    ),
                ],
            ),
        ],
    )
    symlink = repo.upsert_primitive(symlink)
    print(f"Saved: symlink ({symlink.id})")
    return symlink


def seed_user(repo: Repository, os_prim: Primitive) -> Primitive:
    """
    Seed the User primitive at D0–D3.

    A user is the OS-recognized principal of action — the identity against
    which permissions are evaluated and ownership is attributed.
    """
    user = Primitive(
        name="user",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "An entity recognized by the operating system as a principal of "
                                   "action. Has a unique identity within the OS's user management "
                                   "system. Used to attribute ownership, enforce permissions, and "
                                   "establish the security context for running processes.",
                    "identity": "The user's unique name within the OS identity namespace",
                    "uid": "Numeric identifier assigned by the OS for internal reference",
                    "home_directory": "The default filesystem location associated with this user",
                    "process_context": "The security context inherited by processes this user spawns",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Can own files and directories, belong to groups, and run "
                                   "processes. The OS evaluates permissions against the user's "
                                   "identity and group memberships when operations are attempted.",
                    "authenticate": "Prove identity to the OS via password, key, or other mechanism",
                    "own_resource": "Be designated as the owner of a filesystem entity",
                    "spawn_process": "Create a new process running under this user's identity",
                    "elevate": "Request elevated privileges via OS-mediated mechanisms (sudo, setuid)",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=os_prim.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "management": "OS manages user identity, authentication, and identity store",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "User identity must be recognized by the OS. Operations are "
                                   "constrained by permissions granted directly to the user or "
                                   "inherited via group membership. Superuser may bypass most "
                                   "permission constraints.",
                    "single_effective_identity": "A process executes under exactly one effective user identity at a time",
                    "identity_at_creation": "User identity is resolved at process creation, not re-evaluated during execution",
                    "explicit_escalation": "Privilege escalation requires an explicit OS-mediated mechanism",
                    "superuser_designation": "One user identity is designated as superuser with elevated capabilities",
                    "must_exist": "A user must exist in the OS identity store before processes can run under it",
                },
            ),
        ],
    )
    user = repo.upsert_primitive(user)
    print(f"Saved: user ({user.id})")
    return user


def seed_group(repo: Repository, os_prim: Primitive, user: Primitive) -> Primitive:
    """
    Seed the Group primitive at D0–D3.

    A group is a named collection of users through which permissions are
    inherited collectively. The OS resolves group membership when evaluating
    access control.
    """
    group = Primitive(
        name="group",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "A named collection of users recognized by the operating system. "
                                   "Groups aggregate permissions and ownership, allowing multiple users "
                                   "to share access to resources under a common identity.",
                    "name": "The group's unique name within the OS identity namespace",
                    "gid": "Numeric identifier assigned by the OS for internal reference",
                    "members": "The set of user identities that belong to this group",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Can contain multiple users. Permissions granted to a group apply "
                                   "to all its members. The OS evaluates group membership when "
                                   "resolving access control for a user.",
                    "add_member": "Add a user identity to the group's membership",
                    "remove_member": "Remove a user identity from the group's membership",
                    "resolve_members": "Enumerate all users currently in the group",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=os_prim.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "management": "OS manages group identity, membership, and identity store",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.INCLUDES,
                        target_id=user.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Group membership is the mechanism by which permissions are "
                                   "inherited collectively. Membership and identity are managed "
                                   "exclusively by the OS.",
                    "unique_identity": "A group identity must be unique within the OS's identity namespace",
                    "inherited_at_creation": "A process inherits group memberships at creation; changes require a new process context",
                    "membership_limit": "The number of simultaneous group memberships per user is OS-bounded",
                    "must_exist": "A group must exist in the OS identity store before it can be referenced",
                    "privileged_modification": "Membership modification requires privileged access",
                },
            ),
        ],
    )
    group = repo.upsert_primitive(group)
    print(f"Saved: group ({group.id})")
    return group


def seed_process(repo: Repository, os_prim: Primitive) -> Primitive:
    """
    Seed the Process primitive at D0–D3.

    A process is a running instance of a program, managed by the OS.
    It is the runtime context in which all filesystem operations execute,
    carrying user identity, group memberships, and a working directory.
    """
    process = Primitive(
        name="process",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "A running instance of a program, managed by the operating system. "
                                   "Each process has a unique identifier (PID), an effective user "
                                   "identity, group memberships, a working directory, and an environment. "
                                   "It is the runtime context in which all filesystem operations execute.",
                    "pid": "Unique numeric identifier assigned by the OS at creation",
                    "effective_user": "The user identity under which the process operates",
                    "effective_groups": "The set of group memberships inherited at process creation",
                    "working_directory": "The default path context for resolving relative paths",
                    "environment": "Key-value configuration inherited from the parent process or shell",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Can execute programs, perform filesystem operations on behalf "
                                   "of its effective user, spawn child processes, and communicate "
                                   "with other processes. The process inherits its creator's identity "
                                   "and may elevate privileges through OS-mediated mechanisms.",
                    "execute_program": "Load and run an executable file's code",
                    "perform_io": "Read from and write to files, devices, and network sockets",
                    "spawn_child": "Create a new child process inheriting the parent's context",
                    "signal": "Send inter-process signals for communication or termination",
                    "exit": "Terminate the process and release all held resources",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.DEPENDS_ON,
                        target_id=os_prim.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "management": "OS creates, schedules, and terminates processes",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Operates under the authority of its effective user and group "
                                   "memberships. Resource consumption is bounded by OS-enforced limits. "
                                   "A process cannot change its own user identity without an explicit "
                                   "OS-mediated privilege escalation mechanism.",
                    "single_identity": "Operates under exactly one effective user identity at a time",
                    "fixed_groups": "Group memberships are inherited at creation and fixed for the process lifetime",
                    "resource_limits": "Resource consumption is bounded by OS-enforced limits (memory, open files, CPU)",
                    "privilege_escalation": "Cannot change effective user without OS-mediated mechanism (setuid, sudo)",
                    "termination_cleanup": "Termination releases all held resources (file handles, locks, memory)",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.CONSTRAINED_BY,
                        target_id=os_prim.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "description": "Process resource limits and lifecycle are enforced by the OS",
                        },
                    ),
                ],
            ),
        ],
    )
    process = repo.upsert_primitive(process)
    print(f"Saved: process ({process.id})")
    return process


def seed_create(repo: Repository, file: Primitive, directory: Primitive) -> Primitive:
    """
    Seed the Create primitive at D0–D3 with APPLIES_TO relata to File and Directory.
    """
    create = Primitive(
        name="create",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "An operation that brings a new entity into existence where none previously existed.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Accepts a target type, an optional destination context, and optional "
                                   "initial state. Produces a new instance of the target type.",
                    "target_type": "The kind of entity to create (file, directory, symlink)",
                    "destination_context": "The parent location where the new entity will be placed",
                    "initial_state": "Optional content or metadata to populate the new entity with",
                    "result": "A new entity instance at the specified destination",
                },
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The destination context must exist and permit new entities. "
                                   "The target type must be known. The entity must not already exist "
                                   "at the destination unless replacement is explicitly intended.",
                    "destination_exists": "Destination context must exist",
                    "destination_permits": "Destination context must permit creation",
                    "known_type": "Target type must be known",
                    "no_collision": "Entity must not already exist at destination unless replacement is explicit",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "default_destination": "current working directory when no path is specified",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "default_destination": "current working directory when no path is specified",
                        },
                    ),
                ],
            ),
        ],
    )
    create = repo.upsert_primitive(create)
    print(f"Saved: create ({create.id})")
    return create


def seed_read(repo: Repository, file: Primitive) -> Primitive:
    """
    Seed the Read primitive at D0–D3.

    Read is the general operation of retrieving the contents or state of
    an existing entity without modifying it. Domain-specific details
    (partial reads, format interpretation) belong on the relata.
    """
    read = Primitive(
        name="read",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "An operation that retrieves the contents or state of an "
                                   "existing entity without modifying it.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Accepts a target entity and returns its contents or state. "
                                   "Read is inherently non-destructive — the target is unchanged "
                                   "after the operation.",
                    "target_entity": "The entity whose contents or state are to be retrieved",
                    "result": "The contents or state of the target entity",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
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
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The target entity must exist. The actor must have sufficient "
                                   "permission to access it. The entity must be in a state that "
                                   "permits reading.",
                    "target_exists": "Target entity must exist",
                    "read_access": "Actor must have read access to the target",
                    "readable_state": "Target must be in a readable state",
                    "non_destructive": "Read does not modify the entity",
                },
            ),
        ],
    )
    read = repo.upsert_primitive(read)
    print(f"Saved: read ({read.id})")
    return read


def seed_write(repo: Repository, file: Primitive) -> Primitive:
    """
    Seed the Write primitive at D0–D3.

    Write is the general operation of modifying the contents or state
    of an existing entity. Domain-specific details (append vs overwrite,
    encoding) belong on the relata.
    """
    write = Primitive(
        name="write",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "An operation that modifies the contents or state of an "
                                   "existing entity. Unlike create, the entity must already "
                                   "exist. Unlike read, the entity is changed.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Accepts a target entity and new content or state. Produces "
                                   "a modified version of the entity. Write is destructive — "
                                   "previous state may be lost unless explicitly preserved.",
                    "target_entity": "The entity to be modified",
                    "new_content_or_state": "The data or state to apply to the target",
                    "result": "The modified entity with new content or state applied",
                },
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The target entity must exist. The actor must have sufficient "
                                   "permission to modify it. The entity must be in a state that "
                                   "permits modification. Previous state may be irrecoverably lost.",
                    "target_exists": "Target entity must exist",
                    "write_access": "Actor must have write access to the target",
                    "writable_state": "Target must be in a writable state",
                    "destructive": "Write is destructive — previous state may be lost",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "modes": ["overwrite", "append", "insert_at_offset"],
                            "encoding": "content may require encoding (utf-8, binary)",
                            "atomicity": "write may not be atomic — partial writes are possible on failure",
                        },
                    ),
                ],
            ),
        ],
    )
    write = repo.upsert_primitive(write)
    print(f"Saved: write ({write.id})")
    return write


def seed_delete(repo: Repository, file: Primitive, directory: Primitive) -> Primitive:
    """
    Seed the Delete primitive at D0–D4.

    Delete is the general operation of removing an existing entity from
    existence. It is the inverse of create. Includes D4 (IMPLICATIONS)
    to demonstrate cascading consequences.
    """
    delete = Primitive(
        name="delete",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "An operation that removes an existing entity from existence. "
                                   "The inverse of create. After deletion, the entity no longer "
                                   "exists in its context.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Accepts a target entity and removes it. May support "
                                   "recursive deletion for composite entities. Deletion is "
                                   "irreversible unless the system provides recovery mechanisms.",
                    "target_entity": "The entity to be removed from existence",
                    "result": "Confirmation that the entity has been removed",
                },
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The target entity must exist. The actor must have sufficient "
                                   "permission to remove it. Deletion is irreversible by default. "
                                   "Some entities may be protected from deletion.",
                    "target_exists": "Target entity must exist",
                    "delete_access": "Actor must have delete access in the target's context",
                    "irreversible": "Deletion is irreversible by default",
                    "protected_entities": "Some entities may be protected from deletion",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "description": "Removes the file from the filesystem permanently",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "recursive": "directory deletion may require recursive removal of contents",
                            "empty_check": "some systems require the directory to be empty first",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.IMPLICATIONS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Deletion of a filesystem entity has cascading epistemic "
                                   "consequences beyond the immediate removal.",
                    "dangling_symlinks": "Symlinks targeting the deleted entity become unresolvable",
                    "open_handles": "Processes holding file handles to the deleted entity may encounter I/O errors",
                    "recursive_cascade": "Recursive deletion of a directory cascades to all contained entities",
                    "permanent_data_loss": "Data loss is permanent unless backup or journaling mechanisms exist",
                    "executable_unavailability": "Deletion of an executable file prevents future process creation from it",
                    "storage_reclamation": "Freed storage may be reclaimed by the filesystem asynchronously",
                },
            ),
        ],
    )
    delete = repo.upsert_primitive(delete)
    print(f"Saved: delete ({delete.id})")
    return delete


def seed_list(repo: Repository, directory: Primitive) -> Primitive:
    """
    Seed the List primitive at D0–D3.

    List is the general operation of enumerating the contents or members
    of a container. Domain-specific details belong on the relata.
    """
    list_prim = Primitive(
        name="list",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "An operation that enumerates the contents or members of a "
                                   "container entity. Returns a collection of contained entities "
                                   "without modifying the container or its contents.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Accepts a container entity and returns its contents. May "
                                   "support filtering, sorting, and recursive enumeration. "
                                   "List is non-destructive.",
                    "container_entity": "The container whose contents are to be enumerated",
                    "result": "A collection of entities contained within the target",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
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
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The container entity must exist. The actor must have "
                                   "sufficient permission to enumerate its contents. List is "
                                   "non-destructive.",
                    "container_exists": "Container entity must exist",
                    "read_access": "Actor must have read access to the container",
                    "non_destructive": "List does not modify the container or its contents",
                },
            ),
        ],
    )
    list_prim = repo.upsert_primitive(list_prim)
    print(f"Saved: list ({list_prim.id})")
    return list_prim


def seed_move(repo: Repository, file: Primitive, directory: Primitive, path: Primitive) -> Primitive:
    """
    Seed the Move primitive at D0–D3.

    Move is the general operation of relocating an entity from one
    context to another. The entity ceases to exist at the source
    and exists at the destination.
    """
    move = Primitive(
        name="move",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "An operation that relocates an entity from one context to "
                                   "another. The entity ceases to exist at the source location "
                                   "and exists at the destination. Content is preserved.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Accepts a target entity and a destination context. "
                                   "Produces the entity at the destination and removes it "
                                   "from the source. May also rename the entity if the "
                                   "destination implies a new identity.",
                    "target_entity": "The entity to be relocated",
                    "destination_context": "The path or directory where the entity will be moved to",
                    "result": "The entity at its new location, removed from the source",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.REQUIRES,
                        target_id=path.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "source": "path identifies the entity to move",
                            "destination": "path identifies where to move it",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The source entity must exist. The destination context must "
                                   "exist and permit the entity. The actor must have sufficient "
                                   "permission at both source and destination. The entity must "
                                   "not already exist at the destination unless replacement is "
                                   "explicitly intended.",
                    "source_exists": "Source entity must exist",
                    "destination_permits": "Destination context must exist and permit the entity",
                    "dual_access": "Actor must have access at both source and destination",
                    "no_collision": "Entity must not exist at destination unless replacement is explicit",
                    "cross_filesystem": "Move across filesystem boundaries may degrade to copy-then-delete",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "description": "Relocates a file from source to destination; source ceases to exist",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "recursive": "moving a directory moves all of its contents",
                        },
                    ),
                ],
            ),
        ],
    )
    move = repo.upsert_primitive(move)
    print(f"Saved: move ({move.id})")
    return move


def seed_copy(repo: Repository, file: Primitive, directory: Primitive, path: Primitive) -> Primitive:
    """
    Seed the Copy primitive at D0–D3.

    Copy is the general operation of duplicating an entity into a new
    context. Unlike move, the source entity is preserved.
    """
    copy = Primitive(
        name="copy",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "An operation that duplicates an entity, producing a new "
                                   "independent instance with identical contents. The source "
                                   "entity is preserved unchanged.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Accepts a source entity and a destination context. Produces "
                                   "a new entity at the destination with contents identical to "
                                   "the source. The copy is independent — subsequent changes to "
                                   "either do not affect the other.",
                    "source_entity": "The entity to be duplicated",
                    "destination_context": "The path or directory where the copy will be placed",
                    "result": "A new independent entity at the destination with identical contents",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "description": "Duplicates a file; the source is preserved unchanged",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "recursive": "copying a directory copies all of its contents",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.REQUIRES,
                        target_id=path.id,
                        target_depth=DepthLevel.IDENTITY,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "source": "path identifies the entity to copy",
                            "destination": "path identifies where to place the copy",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The source entity must exist. The destination context must "
                                   "exist and have sufficient capacity. The actor must have read "
                                   "access to the source and write access to the destination. "
                                   "The entity must not already exist at the destination unless "
                                   "replacement is explicitly intended.",
                    "source_exists": "Source entity must exist",
                    "destination_capacity": "Destination context must exist and have sufficient capacity",
                    "dual_access": "Actor must have read access at source and write access at destination",
                    "no_collision": "Entity must not exist at destination unless replacement is explicit",
                    "non_destructive": "Copy is non-destructive to the source",
                },
            ),
        ],
    )
    copy = repo.upsert_primitive(copy)
    print(f"Saved: copy ({copy.id})")
    return copy


def seed_modify(repo: Repository, file: Primitive, directory: Primitive, symlink: Primitive) -> Primitive:
    """
    Seed the Modify primitive at D0–D3.

    Modify is the general operation of changing an existing entity's
    attributes or metadata without altering its content. Distinct from
    write (which changes content): modify covers permission changes
    (chmod), ownership changes (chown), and timestamp updates (touch).
    """
    modify = Primitive(
        name="modify",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "An operation that changes the attributes or metadata of an "
                                   "existing entity without altering its content. Distinct from "
                                   "write, which changes content. Modify covers permission changes, "
                                   "ownership transfers, and timestamp updates.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Accepts a target entity, the attribute to change, and the new "
                                   "value. Produces the entity with the updated attribute. The "
                                   "entity's content is unchanged.",
                    "target_entity": "The entity whose attributes are to be changed",
                    "attribute": "The specific metadata attribute to modify (permissions, ownership, timestamps)",
                    "new_value": "The value to set on the target attribute",
                    "result": "The entity with the updated attribute; content is unchanged",
                },
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The target entity must exist. The actor must be the owner "
                                   "or superuser for permission and ownership changes. The "
                                   "attribute must be a valid modifiable property of the entity.",
                    "target_exists": "Target entity must exist",
                    "ownership_required": "Actor must be the entity's owner or superuser for permission changes",
                    "valid_attribute": "The attribute must be a valid modifiable property of the entity",
                    "superuser_for_ownership": "Ownership transfer typically requires superuser privileges",
                    "security_impact": "Modifying permissions changes the entity's security posture",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "permissions": "change read/write/execute bits (chmod)",
                            "ownership": "transfer owner user or group (chown/chgrp)",
                            "timestamps": "update access and modification times (touch)",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=directory.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "permissions": "change read/write/execute bits (chmod)",
                            "ownership": "transfer owner user or group (chown/chgrp)",
                            "timestamps": "update access and modification times (touch)",
                        },
                    ),
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=symlink.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "ownership": "transfer owner user or group (lchown)",
                            "note": "symlink permission bits are typically ignored by the OS",
                        },
                    ),
                ],
            ),
        ],
    )
    modify = repo.upsert_primitive(modify)
    print(f"Saved: modify ({modify.id})")
    return modify


def seed_execute(repo: Repository, file: Primitive) -> Primitive:
    """
    Seed the Execute primitive at D0–D4.

    Execute is the operation that transforms an executable file into a
    running process. It is the only POSIX permission bit without a
    corresponding action in the base set. Includes D4 (IMPLICATIONS)
    for cascading consequences.
    """
    execute = Primitive(
        name="execute",
        provenance=SEED_PROVENANCE,
        depths=[
            Depth(level=DepthLevel.EXISTENCE, provenance=SEED_PROVENANCE),
            Depth(
                level=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "An operation that transforms an executable file into a "
                                   "running process. Unlike read or write, execute changes the "
                                   "category of the entity — from static data to active "
                                   "computation. Requires the execute permission bit, which is "
                                   "distinct from read and write.",
                },
            ),
            Depth(
                level=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Accepts an executable file, an optional argument list, and "
                                   "an optional environment. Produces a new process with the "
                                   "executable's code loaded and running. The new process inherits "
                                   "the invoking user's identity unless the executable has "
                                   "setuid/setgid bits.",
                    "executable_file": "The file containing code to be loaded and executed",
                    "arguments": "Command-line arguments passed to the new process",
                    "environment": "Key-value pairs defining the process's environment variables",
                    "result": "A running process instance with the executable's code loaded",
                },
            ),
            Depth(
                level=DepthLevel.CONSTRAINTS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "The target file must have the execute permission bit set "
                                   "for the effective user. The file must be in a recognized "
                                   "executable format. The actor must have read access to load "
                                   "the file. Setuid/setgid bits may alter the resulting "
                                   "process's identity.",
                    "execute_permission": "Target file must have execute permission for the effective user",
                    "executable_format": "File must be in a recognized executable format (binary or script with shebang)",
                    "read_access": "Actor typically needs read access to load the file into memory",
                    "setuid_setgid": "Setuid/setgid bits may alter the effective identity of the resulting process",
                    "noexec_mount": "The filesystem must not be mounted with the noexec option",
                },
                relata=[
                    Relatum(
                        relation_type=RelationType.APPLIES_TO,
                        target_id=file.id,
                        target_depth=DepthLevel.CAPABILITIES,
                        provenance=SEED_PROVENANCE,
                        metadata={
                            "execute_bit": "requires the execute permission bit, distinct from read/write",
                            "format": "file must contain a recognized executable format",
                        },
                    ),
                ],
            ),
            Depth(
                level=DepthLevel.IMPLICATIONS,
                provenance=SEED_PROVENANCE,
                properties={
                    "description": "Execution transforms static code into active computation, "
                                   "with broad implications for system state.",
                    "filesystem_side_effects": "The resulting process may modify, create, or delete filesystem entities",
                    "resource_consumption": "Resource consumption (memory, CPU, file handles) persists for the process lifetime",
                    "privilege_escalation": "Setuid/setgid executables create privilege escalation vectors",
                    "process_amplification": "The process may spawn child processes, amplifying resource and security impact",
                    "network_exposure": "Network-capable processes may expose the system to external interaction",
                    "inconsistent_state": "A crashed process may leave filesystem entities in an inconsistent state",
                },
            ),
        ],
    )
    execute = repo.upsert_primitive(execute)
    print(f"Saved: execute ({execute.id})")
    return execute


def main(repository: Repository) -> None:
    with repository as repo:
        repo.ensure_constraints()

        # ── Substrates ───────────────────────────────────────────────────
        os_prim = seed_operating_system(repo)
        filesystem = seed_filesystem(repo, os_prim)

        # ── Structural ───────────────────────────────────────────────────
        path = seed_path(repo, filesystem)
        permission = seed_permission(repo, os_prim)
        ownership = seed_ownership(repo, os_prim)

        # ── Entities ─────────────────────────────────────────────────────
        directory = seed_directory(repo, filesystem, path, permission)
        file = seed_file(repo, filesystem, path, permission)
        symlink = seed_symlink(repo, filesystem, path, permission)

        # ── Actors ───────────────────────────────────────────────────────
        user = seed_user(repo, os_prim)
        group = seed_group(repo, os_prim, user)
        process = seed_process(repo, os_prim)

        # ── Actions ──────────────────────────────────────────────────────
        create = seed_create(repo, file, directory)
        read = seed_read(repo, file)
        write = seed_write(repo, file)
        delete = seed_delete(repo, file, directory)
        list_prim = seed_list(repo, directory)
        move = seed_move(repo, file, directory, path)
        copy = seed_copy(repo, file, directory, path)
        modify = seed_modify(repo, file, directory, symlink)
        execute = seed_execute(repo, file)

        # ── Post-creation wiring ─────────────────────────────────────────
        # These relata require primitives that didn't exist when the source
        # was first saved. The pattern: modify in-memory, re-save.

        # filesystem INCLUDES file, directory, symlink
        fs_caps = next(d for d in filesystem.depths if d.level == DepthLevel.CAPABILITIES)
        fs_caps.relata.extend([
            Relatum(
                relation_type=RelationType.INCLUDES,
                target_id=file.id,
                target_depth=DepthLevel.EXISTENCE,
                provenance=SEED_PROVENANCE,
            ),
            Relatum(
                relation_type=RelationType.INCLUDES,
                target_id=directory.id,
                target_depth=DepthLevel.EXISTENCE,
                provenance=SEED_PROVENANCE,
            ),
            Relatum(
                relation_type=RelationType.INCLUDES,
                target_id=symlink.id,
                target_depth=DepthLevel.EXISTENCE,
                provenance=SEED_PROVENANCE,
            ),
        ])
        filesystem = repo.upsert_primitive(filesystem)
        print("Updated: filesystem with INCLUDES relata")

        # permission DEPENDS_ON ownership (eval starts with ownership)
        perm_caps = next(d for d in permission.depths if d.level == DepthLevel.CAPABILITIES)
        perm_caps.relata.append(
            Relatum(
                relation_type=RelationType.DEPENDS_ON,
                target_id=ownership.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission evaluation begins by resolving the actor's "
                                   "ownership relationship to the target entity",
                },
            ),
        )

        # permission APPLIES_TO actors, actions, and entities
        perm_caps.relata.extend([
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=user.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=group.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=process.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs the operations a process may perform "
                                   "based on its effective user and group memberships",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=symlink.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs symlink creation and deletion; "
                                   "dereference permissions are evaluated on the target",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=read.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs whether an actor may perform a read operation",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=write.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs whether an actor may perform a write operation",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=delete.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs whether an actor may perform a delete operation",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=create.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs whether an actor may perform a create operation",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=list_prim.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs whether an actor may enumerate the contents of a directory",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=move.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs whether an actor may relocate an entity across contexts",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=copy.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs whether an actor may duplicate an entity into a new context",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=modify.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs whether an actor may modify the attributes "
                                   "or metadata of an entity",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=execute.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Permission governs whether an actor may execute a file as a program",
                },
            ),
        ])
        permission = repo.upsert_primitive(permission)
        print("Updated: permission with DEPENDS_ON ownership and APPLIES_TO relata")

        # ownership APPLIES_TO entities and actors
        own_caps = next(d for d in ownership.depths if d.level == DepthLevel.CAPABILITIES)
        own_caps.relata.extend([
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=file.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Every file has an owner user and owner group",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=directory.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Every directory has an owner user and owner group",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=symlink.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Every symlink has an owner user and owner group",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=user.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "A user can be the owner of filesystem entities",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=group.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "A group can be the group-owner of filesystem entities",
                },
            ),
        ])
        ownership = repo.upsert_primitive(ownership)
        print("Updated: ownership with APPLIES_TO relata")

        # file, directory, symlink REQUIRES ownership
        for entity_prim in (file, directory, symlink):
            entity_id = next(d for d in entity_prim.depths if d.level == DepthLevel.IDENTITY)
            entity_id.relata.append(
                Relatum(
                    relation_type=RelationType.REQUIRES,
                    target_id=ownership.id,
                    target_depth=DepthLevel.IDENTITY,
                    provenance=SEED_PROVENANCE,
                    metadata={
                        "description": f"Every {entity_prim.name} must have an owner user and group",
                    },
                ),
            )
            entity_prim = repo.upsert_primitive(entity_prim)
        print("Updated: file, directory, symlink with REQUIRES ownership")

        # user INCLUDES process
        user_caps = next(d for d in user.depths if d.level == DepthLevel.CAPABILITIES)
        user_caps.relata.append(
            Relatum(
                relation_type=RelationType.INCLUDES,
                target_id=process.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "A user spawns and owns processes that execute under its identity",
                },
            ),
        )
        user = repo.upsert_primitive(user)
        print("Updated: user with INCLUDES process")

        # CONSTRAINED_BY permission on all action primitives at D3
        for action_prim in (read, write, delete, create, modify):
            action_constraints = next(d for d in action_prim.depths if d.level == DepthLevel.CONSTRAINTS)
            action_constraints.relata.append(
                Relatum(
                    relation_type=RelationType.CONSTRAINED_BY,
                    target_id=permission.id,
                    target_depth=DepthLevel.IDENTITY,
                    provenance=SEED_PROVENANCE,
                    metadata={
                        "description": f"The {action_prim.name} action requires the actor to hold "
                                       f"the corresponding permission on the target",
                    },
                )
            )
            action_prim = repo.upsert_primitive(action_prim)
        print("Updated: read, write, delete, create, modify with CONSTRAINED_BY permission at D3")

        # modify CONSTRAINED_BY ownership (owner authority required)
        modify_constraints = next(d for d in modify.depths if d.level == DepthLevel.CONSTRAINTS)
        modify_constraints.relata.append(
            Relatum(
                relation_type=RelationType.CONSTRAINED_BY,
                target_id=ownership.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Modifying an entity's permissions requires the actor to be "
                                   "the entity's owner or superuser",
                },
            )
        )
        modify = repo.upsert_primitive(modify)
        print("Updated: modify with CONSTRAINED_BY ownership at D3")

        list_constraints = next(d for d in list_prim.depths if d.level == DepthLevel.CONSTRAINTS)
        list_constraints.relata.append(
            Relatum(
                relation_type=RelationType.CONSTRAINED_BY,
                target_id=permission.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "The list action requires read permission on the target directory "
                                   "to enumerate its contents",
                },
            )
        )
        list_prim = repo.upsert_primitive(list_prim)
        print("Updated: list with CONSTRAINED_BY permission at D3")

        move_constraints = next(d for d in move.depths if d.level == DepthLevel.CONSTRAINTS)
        move_constraints.relata.append(
            Relatum(
                relation_type=RelationType.CONSTRAINED_BY,
                target_id=permission.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "The move action requires write permission on both source and "
                                   "destination contexts — the source must permit removal and the "
                                   "destination must permit creation",
                    "source_permission": "write (delete-equivalent) on the source context",
                    "destination_permission": "write (create-equivalent) on the destination context",
                },
            )
        )
        move = repo.upsert_primitive(move)
        print("Updated: move with CONSTRAINED_BY permission at D3")

        copy_constraints = next(d for d in copy.depths if d.level == DepthLevel.CONSTRAINTS)
        copy_constraints.relata.append(
            Relatum(
                relation_type=RelationType.CONSTRAINED_BY,
                target_id=permission.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "The copy action requires read permission on the source and "
                                   "write permission on the destination context",
                    "source_permission": "read on the source entity",
                    "destination_permission": "write (create-equivalent) on the destination context",
                },
            )
        )
        copy = repo.upsert_primitive(copy)
        print("Updated: copy with CONSTRAINED_BY permission at D3")

        execute_constraints = next(d for d in execute.depths if d.level == DepthLevel.CONSTRAINTS)
        execute_constraints.relata.append(
            Relatum(
                relation_type=RelationType.CONSTRAINED_BY,
                target_id=permission.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "The execute action requires execute permission on the target "
                                   "file, which is a distinct permission from read and write",
                    "execute_permission": "the x bit in the POSIX rwx model",
                },
            )
        )
        execute = repo.upsert_primitive(execute)
        print("Updated: execute with CONSTRAINED_BY permission at D3")

        # create, delete APPLIES_TO symlink (destructive — at D3)
        create_constraints = next(d for d in create.depths if d.level == DepthLevel.CONSTRAINTS)
        create_constraints.relata.append(
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=symlink.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Creates a new symlink pointing to a target path",
                },
            )
        )
        create = repo.upsert_primitive(create)
        print("Updated: create with APPLIES_TO symlink at D3")

        delete_constraints = next(d for d in delete.depths if d.level == DepthLevel.CONSTRAINTS)
        delete_constraints.relata.append(
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=symlink.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Removes the symlink without affecting the target entity",
                },
            )
        )

        # delete D4 relata — consequential relationships
        delete_implications = next(d for d in delete.depths if d.level == DepthLevel.IMPLICATIONS)
        delete_implications.relata.extend([
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=symlink.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "consequence": "Symlinks targeting the deleted entity become unresolvable",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=process.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "consequence": "Processes with open handles to the deleted entity may encounter I/O errors",
                },
            ),
        ])
        delete = repo.upsert_primitive(delete)
        print("Updated: delete with APPLIES_TO symlink at D3 and D4 relata")

        # read APPLIES_TO symlink (safe — at D2)
        read_caps = next(d for d in read.depths if d.level == DepthLevel.CAPABILITIES)
        read_caps.relata.append(
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=symlink.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "retrieval": "returns the target path of the symlink, not the target's contents",
                },
            )
        )
        read = repo.upsert_primitive(read)
        print("Updated: read with APPLIES_TO symlink at D2")

        # execute D4 relata — consequential relationships
        execute_implications = next(d for d in execute.depths if d.level == DepthLevel.IMPLICATIONS)
        execute_implications.relata.extend([
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=process.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "consequence": "Execution produces a new process inheriting the invoking user's identity",
                },
            ),
            Relatum(
                relation_type=RelationType.APPLIES_TO,
                target_id=file.id,
                target_depth=DepthLevel.CAPABILITIES,
                provenance=SEED_PROVENANCE,
                metadata={
                    "consequence": "The spawned process may create, modify, or delete files as side effects",
                },
            ),
        ])
        execute = repo.upsert_primitive(execute)
        print("Updated: execute with D4 relata")

        # process CONSTRAINED_BY permission
        proc_constraints = next(d for d in process.depths if d.level == DepthLevel.CONSTRAINTS)
        proc_constraints.relata.append(
            Relatum(
                relation_type=RelationType.CONSTRAINED_BY,
                target_id=permission.id,
                target_depth=DepthLevel.IDENTITY,
                provenance=SEED_PROVENANCE,
                metadata={
                    "description": "Process operations are evaluated against the effective user's permissions",
                    "identity_resolution": "Permission is evaluated using the process's effective user "
                                           "and group memberships",
                },
            )
        )
        process = repo.upsert_primitive(process)
        print("Updated: process with CONSTRAINED_BY permission at D3")

        print("\nDone. Seeded 20 primitives.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Grounded Graph Seeder")
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    args = parser.parse_args()

    repo = Neo4jRepository(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )

    main(repository=repo)
