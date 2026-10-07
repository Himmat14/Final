"""
LaTeX tables of the burn-onset error at EVERY forecast horizon (15, 30, 60, 120, 240, 360 days), written from
outputs/results.json so the report shows how each error grows with the horizon (step 22 writes them).
"""
from pathlib import Path

from .long_run import HORIZONS_DAYS

HEAD = " & ".join(f"{h}\\,d" for h in HORIZONS_DAYS)


def _fmt(x):
    if x is None or x != x:
        return "---"
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def _table(rows, first="Model", extra=None):
    """rows: [(name, {"15d": err, ...}, extra column value or None)]."""
    cols = "l" + ("r" if extra else "") + "r" * len(HORIZONS_DAYS)
    head = f"{first} & " + (f"{extra} & " if extra else "") + HEAD + r"\\"
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head, r"\midrule"]
    for name, errors, value in rows:
        cells = [_fmt(errors.get(f"{h}d")) for h in HORIZONS_DAYS]
        name = name.replace("c_d", "$c_d$")
        lines.append(" & ".join([name] + ([value] if extra else []) + cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_all(results, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    s19 = results.get("report_step19")
    if s19:
        out["horizons_step19.tex"] = _table(
            [(r["scenario"], r["forecast_error_min"], f"{r['period_days']:.4f}") for r in s19["scenarios"]],
            "Scenario", "period [d]")
    s9 = results.get("report_step9")
    if s9:
        out["horizons_step9.tex"] = _table([(n, e, None) for n, e in s9["forecast_error_min_by_horizon"].items()],
                                           "Forecaster")
    s20 = results.get("report_step20")
    if s20:
        for label, key in (("GMM labels", "horizons_step20_gmm.tex"), ("true labels", "horizons_step20_truth.tex")):
            out[key] = _table([(n, v[0], f"{v[1]:.5f}" if v[1] == v[1] else "---")
                               for n, v in s20["forecasts"][label].items()], "Forecaster", "period [d]")
        out["horizons_step20_budget.tex"] = _table(
            [(r["variant"], r["errors"], f"{r['period_days']:.5f}") for r in s20["error_budget"]], "Variant", "period [d]")
        out["horizons_step20_sampling.tex"] = _table(
            [(f"{r['step_s']:g}\\,s" if r["step_s"] < 60 else f"{r['step_s'] / 60:g}\\,min", r["forecast"]["SINDy split (D)"], None)
             for r in s20["sampling"]], "Step (law D)")
    s10 = results.get("report_step10")
    if s10:
        out["horizons_step10_constant.tex"] = _table([(n, e, None) for n, e in s10["mean_abs_error_min_constant"].items()],
                                                     "Forecaster")
        out["horizons_step10_variable.tex"] = _table([(n, e, None) for n, e in s10["mean_abs_error_min_variable"].items()],
                                                     "Forecaster")
    s22 = results.get("report_step22")
    if s22:
        rows = []
        for case in s22["cases"]:
            for model in ("SINDy split (D)", "BINDy split (ARD)", "GP coast curve", "periodic baseline"):
                rows.append((f"{case['label']}: {model}", case["forecasts"][model][0], None))
        out["horizons_step22.tex"] = _table(rows, "Data / forecaster")
    for name, text in out.items():
        (folder / name).write_text(text, encoding="utf-8")
    return sorted(out)
