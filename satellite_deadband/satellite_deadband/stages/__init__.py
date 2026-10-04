"""
Pipeline stages. Each stage module exposes `run(sim, out_dir) -> dict` (`sim` is unused and None: every
stage reads the cached 400-day runs in deadband/long_run.py). It saves its figures into
out_dir/report/stepN_*/ and returns the numbers that end up in results.json.
"""
