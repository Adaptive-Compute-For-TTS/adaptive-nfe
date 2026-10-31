"""Plot a generate.py results CSV: one PNG per path (table + RTF vs NFE), plus eager vs optimized.

The CSV has one row per run (see results/). Times come from whatever GPU ran them; laptop numbers
are rough, so read pass/fail first and RTF as a trend.

Example:
    python scripts/plot_results.py results/generate_base_eager_vs_optimized_2026-10-31.csv
    # -> <csv stem>_eager.png, <csv stem>_optimized.png, <csv stem>_compare.png next to the CSV
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, MUTED, GRID, HEAD, OK = "#fcfcfb", "#0b0b0b", "#52514c", "#8a8980", "#e4e3de", "#f1f0ec", "#0b6b2f"
PROMPT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order, one per prompt
PATH_COLORS = {"eager": "#2a78d6", "optimized": "#eb6834"}
PATH_TITLES = {
    "eager": "eager path (optimize=False): our executor runs every patch's DiT loop",
    "optimized": "optimized path (optimize=True): torch.compile + KV cache, dots.tts's own DiT loop",
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def label(row: dict) -> str:
    return f'{row["prompt_id"]} {row["length_bucket"]}'


def check(v: str) -> str:
    return {"True": "✓ pass", "False": "✗ FAIL"}.get(v, "n/a")


def subtitle(rows: list[dict]) -> str:
    r = rows[0]
    model = r["model"].split("/")[-1]
    n_pass = sum(all(row[k] != "False" for k in ("nfe_fixed_all_patches", "executor_used_all_patches",
                                                  "stock_identical")) for row in rows)
    return f"{model} · bf16 · seed 42 · {r['gpu']} · {n_pass}/{len(rows)} runs pass every check"


def plot_path(rows: list[dict], path: str, out: Path) -> None:
    rows = [r for r in rows if r["path"] == path]
    prompts = sorted({r["prompt_id"] for r in rows})
    fig = plt.figure(figsize=(11, 7.6), facecolor=SURF)
    fig.text(0.04, 0.96, f"generate.py: {PATH_TITLES[path]}", fontsize=13.5, weight="bold")
    fig.text(0.04, 0.927, subtitle(rows), fontsize=10, color=INK2)

    ax_t = fig.add_axes([0.04, 0.47, 0.92, 0.43])
    ax_t.axis("off")
    cols = ["prompt", "NFE", "patches", "audio (s)", "time (s)", "RTF", "VRAM (MB)",
            "NFE fixed\nall patches", "executor\nall patches", "identical\nto stock"]
    cells = [[label(r), r["nfe"], r["num_patches"], f'{float(r["audio_s"]):.2f}', f'{float(r["time_s"]):.1f}',
              f'{float(r["rtf"]):.2f}', f'{float(r["peak_vram_mb"]):.0f}', check(r["nfe_fixed_all_patches"]),
              check(r["executor_used_all_patches"]), check(r["stock_identical"])] for r in rows]
    tbl = ax_t.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.75)
    for (i, j), c in tbl.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_linewidth(0.8)
        if i == 0:
            c.set_facecolor(HEAD)
            c.get_text().set_weight("bold")
            c.set_height(c.get_height() * 1.45)
        else:
            c.set_facecolor(SURF)
            if j >= 7:
                c.get_text().set_color(OK if "pass" in c.get_text().get_text() else MUTED)

    ax = fig.add_axes([0.08, 0.08, 0.64, 0.33], facecolor=SURF)
    ymax = max(float(r["rtf"]) for r in rows) * 1.15
    ends = []
    for color, pid in zip(PROMPT_COLORS, prompts):
        pts = sorted((int(r["nfe"]), float(r["rtf"])) for r in rows if r["prompt_id"] == pid)
        x, y = zip(*pts)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=8, mec=SURF, mew=2, zorder=3)
        ends.append([y[-1], x[-1], label(next(r for r in rows if r["prompt_id"] == pid))])
    # Spread end labels so they don't collide.
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + ymax * 0.07)
    for y, x, text in ends:
        ax.annotate(text, (x, y), xytext=(10, 0), textcoords="offset points", color=INK, fontsize=9.5, va="center")
    nfes = sorted({int(r["nfe"]) for r in rows})
    ax.set_xticks(nfes)
    ax.set_xlim(nfes[0] - 2, nfes[-1] + 8)
    ax.set_ylim(0, ymax)
    ax.axhline(1, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(nfes[-1] + 7.5, 1 - ymax * 0.05, "real-time (RTF 1)", color=MUTED, fontsize=8.5, ha="right", va="top")
    ax.set_xlabel("NFE (denoising steps per patch)")
    ax.set_ylabel("RTF (compute s / audio s)")
    ax.set_title("RTF vs NFE (lower is faster)", loc="left", fontsize=11, weight="bold", pad=8)
    style_axes(ax)

    executor_note = ("• executor: Yulchan's executor\n  decoded every patch\n" if path == "eager"
                     else "• executor: n/a, dots.tts's own\n  fast loop (scheduler sets NFE)\n")
    fig.text(0.77, 0.38, "Checks per run", weight="bold", fontsize=10, va="top")
    fig.text(0.77, 0.35,
             "• NFE fixed: every patch got\n  the chosen NFE\n" + executor_note +
             "• identical: audio bit-for-bit\n  equal to stock dots.tts on\n  the same path\n\n"
             "Laptop GPU timings are rough;\nread pass/fail first.", fontsize=8.8, color=INK2, va="top")
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)


def plot_compare(rows: list[dict], out: Path) -> None:
    nfes = sorted({int(r["nfe"]) for r in rows})
    agg = {}  # path -> [aggregate RTF per NFE] = total compute time / total audio over all prompts
    for path in PATH_COLORS:
        agg[path] = []
        for n in nfes:
            rs = [r for r in rows if r["path"] == path and int(r["nfe"]) == n]
            agg[path].append(sum(float(r["time_s"]) for r in rs) / sum(float(r["audio_s"]) for r in rs))

    fig = plt.figure(figsize=(11, 5.6), facecolor=SURF)
    fig.text(0.04, 0.94, "Eager (our executor) vs optimized (dots.tts fast path): RTF vs NFE", fontsize=13.5,
             weight="bold")
    fig.text(0.04, 0.895, subtitle(rows) + " · RTF = total time / total audio over all prompts",
             fontsize=10, color=INK2)

    ax = fig.add_axes([0.08, 0.12, 0.46, 0.68], facecolor=SURF)
    ymax = max(max(v) for v in agg.values()) * 1.15
    for path, color in PATH_COLORS.items():
        ax.plot(nfes, agg[path], color=color, lw=2, marker="o", ms=8, mec=SURF, mew=2, zorder=3)
        ax.annotate(path, (nfes[-1], agg[path][-1]), xytext=(10, 0), textcoords="offset points", color=INK,
                    fontsize=10, va="center")
        for n, v in zip(nfes, agg[path]):
            ax.annotate(f"{v:.2f}", (n, v), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK2)
    ax.set_xticks(nfes)
    ax.set_xlim(nfes[0] - 2, nfes[-1] + 8)
    ax.set_ylim(0, ymax)
    ax.axhline(1, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(nfes[-1] + 7.5, 1 - ymax * 0.03, "real-time", color=MUTED, fontsize=8.5, ha="right", va="top")
    ax.set_xlabel("NFE (denoising steps per patch)")
    ax.set_ylabel("RTF (compute s / audio s)")
    style_axes(ax)

    ax_t = fig.add_axes([0.6, 0.3, 0.37, 0.45])
    ax_t.axis("off")
    cells = [[str(n), f"{e:.2f}", f"{o:.2f}", f"{e / o:.1f}×", f"{e / agg['eager'][0]:.1f}×",
              f"{o / agg['optimized'][0]:.1f}×"]
             for n, e, o in zip(nfes, agg["eager"], agg["optimized"])]
    tbl = ax_t.table(cellText=cells, colLabels=["NFE", "eager\nRTF", "opt.\nRTF", "opt.\nspeedup",
                                                f"eager\nvs NFE {nfes[0]}", f"opt.\nvs NFE {nfes[0]}"],
                     loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.9)
    for (i, j), c in tbl.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_linewidth(0.8)
        c.set_facecolor(HEAD if i == 0 else SURF)
        if i == 0:
            c.get_text().set_weight("bold")
            c.set_height(c.get_height() * 1.5)
    fig.text(0.6, 0.25,
             "Both paths: more NFE costs more time. The optimized path\nis faster at every NFE, and NFE is a bigger share "
             "of its\ntime (NFE 4→32 costs more, relatively), so cutting NFE\nmatters more there. "
             "Laptop GPU timings are rough.",
             fontsize=8.8, color=INK2, va="top")
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    a = p.parse_args()
    src = Path(a.csv)
    rows = list(csv.DictReader(src.open()))
    for path in sorted({r["path"] for r in rows}):
        out = src.with_name(f"{src.stem}_{path}.png")
        plot_path(rows, path, out)
        print(out)
    if {r["path"] for r in rows} >= set(PATH_COLORS):
        out = src.with_name(f"{src.stem}_compare.png")
        plot_compare(rows, out)
        print(out)


if __name__ == "__main__":
    main()
