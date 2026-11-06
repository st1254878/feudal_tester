"""
Plot the goal-ablation results from reports_cont/goal_ablation/*.json (run goal_ablation.py first).
Writes reports_cont/halfcheetah_goal_ablation.png and reports_cont/ant_goal_ablation.png.

Usage: python reports_cont/plot_goal_ablation.py
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "goal_ablation")

# Reference palette, light mode
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, LEGEND_FILL = "#e1e0d9", "#c3c2b7", "#d6d5cd"
MODELS = [("fun", "FuN", "#2a78d6"), ("cat", "FeUdalCat", "#eb6834")]
CONDITIONS = [("real", "Real\ngoals"), ("frozen", "Frozen\n(1st goal)"), ("random10", "Random,\nnew every\n10 steps"),
              ("random1", "Random,\nnew every\nstep"), ("zero", "Zero\ngoal")]


def pct(v):
    return f"{v:+.0f}%".replace("-", "−")


def verdict(c, still):
    """Two facts (frozen and zero goal vs real) and the conclusion they support."""
    real, frozen, zero = (np.mean(c[k]["returns"]) for k in ("real", "frozen", "zero"))
    f_chg, z_chg = (frozen - real) / abs(real) * 100, (zero - real) / abs(real) * 100
    moves_without_goal = zero > 0.3 * real and zero > 1.3 * still
    if not moves_without_goal:
        concl = ("→ needs a goal, but a constant one is enough" if abs(f_chg) < 10
                 else "→ needs the goal, and its changes matter")
    else:
        concl = ("→ still moves with no goal; the 1st goal hurts more than none" if frozen < 0.95 * zero
                 else "→ still moves with no goal; the real goal adds the rest")
    return f"Frozen goal {pct(f_chg)} · zero goal {pct(z_chg)}\n{concl}"


def panel(ax, res, name, color, ylim, ref, show_ref_label):
    rng = np.random.default_rng(0)
    span = ylim[1] - ylim[0]
    for i, (cond, _) in enumerate(CONDITIONS):
        r = np.array(res["conditions"][cond]["returns"])
        mean, med = r.mean(), np.median(r)
        ax.bar(i, mean, width=0.62, color=color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot([i - 0.31, i + 0.31], [mean, mean], color=color, linewidth=3, solid_capstyle="butt", zorder=3)
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(r)), r, s=11, color=color, alpha=0.6, linewidths=0, zorder=4)
        ax.plot([i - 0.18, i + 0.18], [med, med], color=INK, linewidth=2.2, solid_capstyle="butt", zorder=5)
        ax.text(i, max(mean, r.max(), 0) + 0.025 * span, f"{int(round(mean)):,}".replace("-", "−"), ha="center",
                va="bottom", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.98, 1.0, verdict(res["conditions"], ref[0] if ref else 0.0), transform=ax.transAxes, ha="right",
            va="top", fontsize=10.5, color=INK_2, linespacing=1.5)
    if ref is not None:
        ax.axhline(ref[0], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        if show_ref_label:
            ax.text(-0.55, ref[0] - 0.012 * span, ref[1], ha="left", va="top", fontsize=8.5, color=MUTED, zorder=6)
    ax.set_title(name, loc="left", fontsize=15, fontweight="bold", color=color, pad=14)
    ax.set_facecolor(SURFACE)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([lab for _, lab in CONDITIONS], fontsize=10, color=INK_2)
    ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", colors=MUTED, labelsize=10, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


def save(fig, path):
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    for _ in range(20):  # a viewer or file scanner can hold the old PNG open for a moment
        try:
            os.replace(tmp, path)
            break
        except OSError:
            time.sleep(0.5)
    else:
        raise OSError(f"could not replace {path}; new figure left at {tmp}")
    print("wrote", path)


def figure(env_key, env_title, out_name, ref=None):
    res = {m: json.load(open(os.path.join(DATA, f"{env_key}_{m}.json"), encoding="utf-8")) for m, _, _ in MODELS}
    allv = np.concatenate([np.array(v["returns"]) for r in res.values() for v in r["conditions"].values()])
    lo = min(0.0, allv.min())
    hi = allv.max()
    pad = 0.26 * (hi - lo)  # headroom for the bar labels and the verdict text
    ylim = (lo - (0.04 * (hi - lo) if lo < 0 else 0), hi + pad)
    r0 = next(iter(res.values()))
    n_ep, step = len(r0["seeds"]), r0["step"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.9), sharey=True, facecolor=SURFACE)
    for k, (ax, (m, name, color)) in enumerate(zip(axes, MODELS)):
        panel(ax, res[m], name, color, ylim, ref, show_ref_label=(k == 0))
    axes[0].set_ylabel("Episode return", fontsize=12, color=INK_2)
    fig.suptitle(f"{env_title} — does the Worker need the Manager’s goals?", x=0.012, y=0.975, ha="left",
                 fontsize=17, fontweight="bold", color=INK)
    fig.text(0.012, 0.905, f"Final {step/1e6:.0f}M checkpoints (V4 code) with the Manager’s goal replaced. "
                           f"{n_ep} episodes per bar, same env seeds in every bar, deterministic actions.",
             fontsize=11, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LEGEND_FILL),
               plt.Line2D([], [], color=INK, linewidth=2.2),
               plt.Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=5)]
    fig.legend(handles, ["Mean (bar height and number)", "Median", "One episode"], loc="lower left", ncol=3,
               frameon=False, fontsize=10.5, labelcolor=INK_2, bbox_to_anchor=(0.005, 0.06))
    fig.text(0.012, 0.012, "Mean = average of the episode returns. Median = the middle return (half the episodes "
                           "score more, half less).\nThe Manager still runs; only the goal passed to the Worker is "
                           "replaced. Random goals are unit vectors, like real ones.", fontsize=8.5, color=MUTED,
             linespacing=1.4)
    fig.tight_layout(rect=(0, 0.11, 1, 0.9), w_pad=3)
    save(fig, os.path.join(HERE, out_name))


figure("cheetah", "HalfCheetah", "halfcheetah_goal_ablation.png")
figure("ant", "Ant", "ant_goal_ablation.png", ref=(1000, "≈1,000 = standing still"))
