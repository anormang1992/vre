"""
Clear all primitives and relationships from a VRE repository.

Can be run standalone or imported by seed scripts to ensure a clean slate.

Run: python scripts/clear_graph.py
"""
import argparse

from scripts import add_backend_args, make_repository
from vre.core.backends import Repository


def clear_graph(repo: Repository) -> int:
    """
    Delete every Primitive and its relationships. Returns the count deleted.
    """
    return repo.clear()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clear all VRE graph data")
    add_backend_args(parser)
    args = parser.parse_args()

    repo = make_repository(args)
    with repo:
        deleted = clear_graph(repo)
        print(f"Cleared {deleted} primitive(s) from the graph.")
