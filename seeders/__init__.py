"""
Domain seed scripts for the Volute Reasoning Engine.

Each module in this package authors a self-contained epistemic domain by
upserting primitives into a Neo4j graph. Seeders are idempotent (via
PrimitiveRepository.upsert_primitive) and do not clear the graph — multiple
domains can be installed side by side.

Add a new domain by dropping a module here that exposes a `main(repository)`
entry point.
"""
