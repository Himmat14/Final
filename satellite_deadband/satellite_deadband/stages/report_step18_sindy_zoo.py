"""
Report step 18: the SINDy equation zoo (deadband/sindy_zoo.py): 10 candidate libraries for each of three
datasets (full-physics LEO with constant drag, LEO with space-weather drag, GEO longitude), compared on
held-out error, sparsity, BIC and free-running forecasts, with the coefficients each library learns.

Figures (outputs/report/step18_sindy_zoo/)
    step18_scores_<dataset>     per library: test error, active terms, BIC and forecast error
    step18_pareto               forecast error vs number of active terms, all datasets
    step18_coefficients         the learned equation of the best few libraries per dataset
    step18_summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import sindy_zoo as Z
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST
from .report_common import panel_label, report_style, save, step_dir

DATASET_COLORS = (NAVY, RUST, GREEN)


def _short(name):
    return name.split(". ", 1)[1] if ". " in name else name


def _draw_scores(axes, rows, data):
    names = [_short(r["library"]) for r in rows]
    y = np.arange(len(rows))
    metrics = (("coast_test_nrmse" if not data.name.startswith("GEO") else "test_nrmse",
                "test error (normalised RMSE)", NAVY, True),
               ("active", "active terms", GREY, False), ("bic", "BIC (train, lower = better)", AMBER, False),
               ("forecast", rows[0]["forecast_unit"], RUST, True))
    for ax, (key, label, color, log) in zip(axes, metrics):
        values = np.array([r[key] for r in rows], dtype=float)
        finite = np.isfinite(values)
        bars = ax.barh(y[finite], values[finite], color=color)
        for k in np.flatnonzero(~finite):
            ax.text(0.02, k, "no usable forecast (law never decays)", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7, color=RUST)
        if key != "active" and finite.any():                 # outline the best library
            best = int(np.nanargmin(np.where(finite, values, np.nan)))
            bars[int(np.sum(finite[:best]))].set(edgecolor="black", linewidth=2)
        ax.set(xscale="log" if log else "linear", xlabel=label, title=label.split(" [")[0],
               yticks=y, ylim=(len(rows) - 0.5, -0.5))          # first library at the top
        ax.set_yticklabels(names if ax is axes[0] else [str(k + 1) for k in range(len(rows))], fontsize=8)
        ax.set_ylabel("library" if ax is axes[0] else "library (number)")
        if key == "forecast":                                  # classic STLSQ for comparison (hatched)
            classic = np.array([r.get("classic_forecast", np.nan) for r in rows], dtype=float)
            ok = np.isfinite(classic)
            ax.barh(y[ok] + 0.3, classic[ok], height=0.3, color="white", edgecolor=RUST, hatch="///",
                    label="classic (relative) STLSQ")
            bars.set_label("significance STLSQ")
            ax.legend(fontsize=7, loc="lower right")


def _draw_pareto(ax, results):
    for (name, (data, rows)), color in zip(results.items(), DATASET_COLORS):
        values = np.array([r["forecast"] for r in rows], dtype=float)
        best = np.nanmin(values)
        for r in rows:
            if np.isfinite(r["forecast"]):
                ax.plot(r["active"], r["forecast"] / best, "o", ms=7, color=color)
                ax.annotate(r["library"].split(".")[0], (r["active"], r["forecast"] / best), fontsize=7,
                            textcoords="offset points", xytext=(4, 2))
        ax.plot([], [], "o", color=color, label=name)
    ax.set(yscale="log", xlabel="active terms in the learned equation", ylabel="forecast error / best library's error",
           title="Simplicity vs forecast skill (numbers = library)")
    ax.legend(fontsize=7)


def _draw_coefficients(ax, results):
    lines = []
    for name, (data, rows) in results.items():
        lines.append(f"{name}  [{data.unit}]")
        ranked = sorted((r for r in rows if np.isfinite(r["forecast"])), key=lambda r: r["forecast"])[:3]
        for r in ranked:
            terms = " ".join(f"{c:+.4g} {n}" for c, n in zip(r["coefficients"], r["names"]) if c != 0)
            extra = f"   A={r['recovered'][0]:.5f}, lambda_s={r['recovered'][1]:.2f}" if r.get("recovered") and r["recovered"][0] else ""
            lines.append(f"  {_short(r['library'])[:34]:<34} {terms[:110]}{extra}")
            lines.append(f"  {'':<34} classic STLSQ keeps only: {', '.join(r['classic_names_kept']) or 'nothing'}")
        lines.append("")
    ax.axis("off")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7.5)
    ax.set_title("The best three equations per dataset (by forecast error)")


def run(sim, out_dir):
    folder = step_dir(out_dir, "step18_sindy_zoo")
    results = Z.run_zoo()
    with report_style():
        for k, (name, (data, rows)) in enumerate(results.items()):
            fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
            _draw_scores(axes, rows, data)
            save(fig, folder, f"step18_scores_{k + 1}.png", suptitle=f"SINDy libraries on: {name}")
        fig, ax = plt.subplots(figsize=(9, 5.5)); _draw_pareto(ax, results); save(fig, folder, "step18_pareto.png")
        fig, ax = plt.subplots(figsize=(16, 6)); _draw_coefficients(ax, results); save(fig, folder, "step18_coefficients.png")
        fig = plt.figure(figsize=(20, 11))
        grid = fig.add_gridspec(2, 2)
        _draw_pareto(fig.add_subplot(grid[0, 0]), results)
        _draw_coefficients(fig.add_subplot(grid[0, 1]), results)
        data, rows = list(results.values())[1]
        axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        names = [_short(r["library"]) for r in rows]
        axes[0].barh(names, [r["coast_test_nrmse"] for r in rows], color=NAVY)
        axes[0].set(xscale="log", xlabel="coast test error (normalised RMSE, log)", ylabel="library",
                    title="LEO space-weather data: fit to held-out data")
        axes[0].invert_yaxis()
        fc = np.array([r["forecast"] for r in rows], dtype=float)
        axes[1].barh(names, np.where(np.isfinite(fc), fc, np.nan), color=RUST)
        axes[1].set(xscale="log", xlabel="mean |burn-onset error| over 120 days [min]", ylabel="library",
                    title="... and in a free-running forecast")
        axes[1].invert_yaxis()
        for ax, letter in zip(fig.axes, "abcd"):
            panel_label(ax, letter)
        save(fig, folder, "step18_summary.png", suptitle="Step 18: many candidate equations for the satellite's behaviour")
    out = {}
    for name, (data, rows) in results.items():
        out[name] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k != "library_fn"}
                     for r in rows]
    return dict(report_step18=out)
