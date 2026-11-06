"""
Eval curves of the final-code runs that ran to completion:
  reports/halfcheetah_eval.png  HalfCheetah-v4, V4 code, 0-10M  (FuN, FeUdalCat, Worker-only)
  reports/ant_eval.png          Ant-v4, V4 code, 0-20M          (FuN, FeUdalCat, Worker-only)
  reports/antmaze_eval.png      AntMaze_UMazeDense-v4 long runs (FuN to 87.5M incl. resume, FeUdalCat to 73.5M)

Usage: python reports/plot_evals.py
"""
import glob
import os
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.dirname(os.path.abspath(__file__))

# Reference palette, light mode: categorical slots 1-3, chrome and ink
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = {"FuN": "#2a78d6", "FeUdalCat": "#eb6834", "Worker-only": "#1baf7a"}


def eval_curve(*run_names, tag="eval/mean_reward", scale=1.0):
    """Eval scalar vs. step; several run folders are concatenated (used for resumed runs)."""
    steps, vals = [], []
    for name in run_names:
        for ev in glob.glob(os.path.join(ROOT, "runs_continuous", name, "events.out.tfevents.*")):
            ea = event_accumulator.EventAccumulator(ev, size_guidance={"scalars": 0})
            ea.Reload()
            if tag not in ea.Tags()["scalars"]:
                continue
            for e in ea.Scalars(tag):
                steps.append(e.step)
                vals.append(e.value * scale)
    order = np.argsort(steps)
    return np.array(steps)[order] / 1e6, np.array(vals)[order]


def smooth(x, y, n_evals=None, window_m=None):
    """Trailing rolling mean over the last n_evals points, or over the last window_m million steps."""
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - n_evals + 1) if n_evals else np.searchsorted(x, x[i] - window_m, side="right")
        out[i] = y[lo: i + 1].mean()
    return out


def style_axes(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlabel("Environment steps (millions)", color=INK_2, fontsize=9)


def plot_panel(ax, panel, smoothing, fmt):
    style_axes(ax, panel["title"])
    ylim = panel["ylim"]
    ends = []
    for label, (x, y) in panel["curves"].items():
        c = COLORS[label]
        ax.plot(x, y, color=c, linewidth=0.8, alpha=0.22)
        ys = smooth(x, y, **smoothing)
        ax.plot(x, ys, color=c, linewidth=2, solid_capstyle="round")
        ends.append([ys[-1], x[-1], label])
    # End labels, pushed apart vertically so they never overlap
    min_gap = 0.06 * (ylim[1] - ylim[0])
    ends.sort(key=lambda e: e[0])
    label_y = [e[0] for e in ends]
    for i in range(1, len(label_y)):
        label_y[i] = max(label_y[i], label_y[i - 1] + min_gap)
    for (y_end, x_end, label), y_lab in zip(ends, label_y):
        ax.annotate(f"{label}  {fmt(y_end)}", (x_end, y_lab), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(*panel["xlim"])
    ax.set_ylim(*ylim)
    ax.set_ylabel(panel["ylabel"], color=INK_2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(v)))
    if panel.get("xticks") is not None:
        ax.set_xticks(panel["xticks"])
    if panel.get("ref"):
        y_ref, text = panel["ref"]
        ax.axhline(y_ref, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(text, (0.99, y_ref), xycoords=("axes fraction", "data"), xytext=(0, -5),
                    textcoords="offset points", fontsize=8, color=MUTED, ha="right", va="top")
    if panel.get("vline"):
        x_v, text = panel["vline"]
        ax.axvline(x_v, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(text, (x_v, 0.97), xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=MUTED, ha="left", va="top")


def figure(panels, suptitle, note, path, labels, smoothing, fmts):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n + 0.6, 4.6), facecolor=SURFACE, squeeze=False)
    for ax, panel, fmt in zip(axes[0], panels, fmts):
        plot_panel(ax, panel, smoothing, fmt)
    handles = [plt.Line2D([], [], color=COLORS[l], linewidth=2) for l in labels]
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(suptitle, x=0.01, y=0.995, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    note = textwrap.fill(note, width=125 if n == 1 else 250)
    n_lines = note.count("\n") + 1
    fig.text(0.01, 0.008, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035 * n_lines + 0.015, 0.96, 0.95))
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    for attempt in range(20):  # a viewer or file scanner can hold the old PNG open for a moment
        try:
            os.replace(tmp, path)
            break
        except OSError:
            time.sleep(0.5)
    else:
        raise OSError(f"could not replace {path}; new figure left at {tmp}")
    print("wrote", path)


ret_fmt = lambda v: f"{v:,.0f}"
pct_fmt = lambda v: f"{v:.0f}%"
mujoco_note = ("Thick line: rolling mean of the last 10 evals. Faint line: each eval (mean of 20 deterministic "
               "episodes). Number at the end: last smoothed value.")

figure(
    [dict(title="V4 code, trained from scratch, 0–10M steps", ylabel="Eval return",
          curves={"FuN": eval_curve("20261018_021526_cont_cheetah_fun_10m"),
                  "FeUdalCat": eval_curve("20261018_021543_cont_cheetah_cat_10m"),
                  "Worker-only": eval_curve("20261019_231417_cont_cheetah_worker_only_10m")},
          xlim=(0, 11.6), ylim=(-500, 5500), xticks=np.arange(0, 10.1, 2))],
    "HalfCheetah-v4 — eval return", mujoco_note, os.path.join(OUT, "halfcheetah_eval.png"),
    labels=["FuN", "FeUdalCat", "Worker-only"], smoothing={"n_evals": 10}, fmts=[ret_fmt],
)

figure(
    [dict(title="V4 code, trained from scratch, 0–20M steps", ylabel="Eval return",
          curves={"FuN": eval_curve("20261018_114728_cont_ant_fun_20m"),
                  "FeUdalCat": eval_curve("20261018_114744_cont_ant_cat_20m"),
                  "Worker-only": eval_curve("20261019_231437_cont_ant_worker_only_20m")},
          xlim=(0, 23.2), ylim=(0, 5500), xticks=np.arange(0, 20.1, 5),
          ref=(1000, "dashed: ≈1,000 = standing still (survival bonus only)"))],
    "Ant-v4 — eval return", mujoco_note, os.path.join(OUT, "ant_eval.png"),
    labels=["FuN", "FeUdalCat", "Worker-only"], smoothing={"n_evals": 10}, fmts=[ret_fmt],
)

MAZE_FUN = ("20261026_030411_cont_antmaze_fun_c50_bonus50_100m", "20261031_001629_cont_antmaze_fun_c50_bonus50_100m")
MAZE_CAT = ("20261026_030433_cont_antmaze_cat_c50_bonus50_100m",)
maze_common = dict(xlim=(20, 94), xticks=np.arange(20, 90.1, 10), vline=(73.7, "FuN resumed"))
figure(
    [dict(title="Eval success rate", ylabel="Success rate (20 episodes)",
          curves={"FuN": eval_curve(*MAZE_FUN, tag="eval/success_rate", scale=100),
                  "FeUdalCat": eval_curve(*MAZE_CAT, tag="eval/success_rate", scale=100)},
          ylim=(0, 40), **maze_common),
     dict(title="Eval return (exp(−distance) per step + 50 at the goal)", ylabel="Eval return",
          curves={"FuN": eval_curve(*MAZE_FUN), "FeUdalCat": eval_curve(*MAZE_CAT)},
          ylim=(0, 100), **maze_common)],
    "AntMaze_UMazeDense-v4 — long runs (dense reward + 50 success bonus)",
    "Thick line: rolling mean over the last 2M steps. Faint line: each eval (20 deterministic episodes). "
    "Both runs continue from a 0–20M run whose logs no longer exist, so curves start at 20M.",
    os.path.join(OUT, "antmaze_eval.png"),
    labels=["FuN", "FeUdalCat"], smoothing={"window_m": 2.0}, fmts=[pct_fmt, ret_fmt],
)
