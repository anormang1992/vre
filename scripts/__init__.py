import argparse

from vre.core.backends import Repository


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["neo4j", "sqlite"],
        default="sqlite",
        help="Persistence backend (default: sqlite)",
    )

    neo4j = parser.add_argument_group("neo4j", "Options for --backend neo4j")
    neo4j.add_argument("--neo4j-uri", default="neo4j://localhost:7687")
    neo4j.add_argument("--neo4j-user", default="neo4j")
    neo4j.add_argument("--neo4j-password", default="password")

    sqlite = parser.add_argument_group("sqlite", "Options for --backend sqlite")
    sqlite.add_argument(
        "--sqlite-path",
        default=None,
        help="Database path (default: ~/.vre/graph.db, use ':memory:' for in-memory)",
    )


def make_repository(args: argparse.Namespace) -> Repository:
    if args.backend == "sqlite":
        from vre.core.backends import SQLiteRepository
        return SQLiteRepository(args.sqlite_path)
    from vre.core.backends import Neo4jRepository
    return Neo4jRepository(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
