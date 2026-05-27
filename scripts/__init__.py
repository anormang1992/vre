import argparse

from vre.core.backends import Repository


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["neo4j", "sqlite"],
        default="neo4j",
        help="Persistence backend (default: neo4j)",
    )
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument(
        "--sqlite-path",
        default=None,
        help="SQLite database path (default: ~/.vre/graph.db, use ':memory:' for in-memory)",
    )


def make_repository(args: argparse.Namespace) -> Repository:
    if args.backend == "sqlite":
        from vre.core.backends import SQLiteRepository
        return SQLiteRepository(args.sqlite_path)
    from vre.core.backends import Neo4jRepository
    return Neo4jRepository(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
