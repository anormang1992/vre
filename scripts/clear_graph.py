"""
Clear all primitives and relationships from a VRE repository.

Can be run standalone or imported by seed scripts to ensure a clean slate.

Run: python scripts/clear_graph.py
"""
import argparse

from vre.core.backends import Neo4jRepository, Repository


def clear_graph(repo: Repository) -> int:
    """
    Delete every Primitive and its relationships. Returns the count deleted.
    """
    return repo.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear all VRE graph data")
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    args = parser.parse_args()

    repo = Neo4jRepository(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )

    with repo:
        deleted = clear_graph(repo)
        print(f"Cleared {deleted} primitive(s) from the graph.")
