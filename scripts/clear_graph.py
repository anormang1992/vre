"""
Clear all primitives and relationships from the VRE Neo4j graph.

Can be run standalone or imported by seed scripts to ensure a clean slate.

Run: poetry run python scripts/clear_graph.py
"""
import argparse
from typing import LiteralString, cast

from vre.core.graph import PrimitiveRepository


def clear_graph(repo: PrimitiveRepository) -> int:
    """
    Delete every Primitive node and its relationships. Returns the count deleted.
    """
    with repo._driver.session(database=repo._database) as session:
        result = session.run(
            cast(
                LiteralString,
                "MATCH (p:Primitive) DETACH DELETE p RETURN count(p) AS deleted",
            ),
        ).single()
        return result["deleted"] if result else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear all VRE graph data")
    parser.add_argument("--neo4j-uri", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    args = parser.parse_args()

    repo = PrimitiveRepository(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )

    with repo:
        deleted = clear_graph(repo)
        print(f"Cleared {deleted} primitive(s) from the graph.")
