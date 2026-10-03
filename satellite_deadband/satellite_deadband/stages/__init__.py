"""
Pipeline stages. Each stage module exposes `run(sim, out_dir) -> dict`: it runs one
workstream from `deadband/`, saves its figures into `out_dir`, and returns the numbers
that end up in results.json.
"""
