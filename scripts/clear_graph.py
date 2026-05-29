"""
Clear all primitives and relata from a VRE repository.

Run: python scripts/clear_graph.py
"""
import argparse

from scripts import add_backend_args, make_repository


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear all VRE graph data")
    add_backend_args(parser)
    args = parser.parse_args()

    repo = make_repository(args)
    with repo:
        deleted = repo.clear()
        print(f"Cleared {deleted} primitive(s) from the graph.")
