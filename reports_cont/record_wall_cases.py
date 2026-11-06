"""
AntMaze "goal behind the wall" test: start and goal are on opposite arms of the U, so the straight line
between them crosses the 4 m thick middle wall and the ant has to go around through the right corridor.

Both AntMaze checkpoints were trained on the base env's Euclidean dense reward exp(-d), which is highest
right against the wall on the wrong side (d ~ 6-8 m) until the ant commits to the detour. This script
checks whether the agent goes around or sits on the wall face ("wall farming").

For each model and each start/goal pair it runs N noisy episodes (+-1 m start/goal noise, deterministic
actions), classifies each one, writes a trajectory plot, and records GIFs:
  - the most common outcome for that pair (first seed with that outcome, not cherry-picked)
  - plus the fastest success, if any episode succeeded and success is not already the most common outcome

Outputs: reports/gifs/wall/<model>_<pair>_<outcome>.gif, reports/antmaze_wall_trajectories.png
Usage: python reports/record_wall_cases.py --device cuda [--episodes 10] [--no-gif]
"""
import argparse
import os
import sys
from collections import Counter

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_peaks import CK, MODEL_NAME, load_model, make_env, setup_camera  # noqa: E402

ENV_ID = "AntMaze_UMazeDense-v4"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gifs", "wall")
PLOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "antmaze_wall_trajectories.png")
MODELS = [
    ("antmaze_fun", "20261031_001629_cont_antmaze_fun_c50_bonus50_100m/model_step_81716800.pt", "fun"),
    ("antmaze_feudal_cat", "20261026_030433_cont_antmaze_cat_c50_bonus50_100m/model_best.pt", "feudal_cat"),
]
# U-maze cells (row, col); row 1 = top arm, row 3 = bottom arm, the wall (2,1)-(2,2) sits between them,
# the only connection is the right corridor (1,3)-(2,3)-(3,3).
PAIRS = [
    ("top_left_to_bottom_left", (1, 1), (3, 1), "start top-left -> goal bottom-left, behind wall"),
    ("bottom_left_to_top_left", (3, 1), (1, 1), "start bottom-left -> goal top-left, behind wall"),
    ("top_mid_to_bottom_mid", (1, 2), (3, 2), "start top-middle -> goal bottom-middle, behind wall"),
    ("bottom_mid_to_top_mid", (3, 2), (1, 2), "start bottom-middle -> goal top-middle, behind wall"),
]
SCALE, CENTER = 4.0, 10.0   # maze_size_scaling and map center (5x5 map * 4 m / 2)
MAZE = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
FRAME_SKIP = 3              # every 3rd frame at 20 fps -> about 3x real time (sim dt = 0.05 s)
FRAME_MS = 50
AFTER_SUCCESS = 40
OUTCOMES = ["reached goal", "went around, missed goal", "into corridor, turned back", "stuck behind wall"]
TAGS = {OUTCOMES[0]: "reached", OUTCOMES[1]: "around_missed", OUTCOMES[2]: "corridor_only", OUTCOMES[3]: "stuck"}


def xy_to_cell(x, y):
    return int(np.floor((CENTER - y) / SCALE)), int(np.floor((x + CENTER) / SCALE))


def classify(traj, start_cell, success_step):
    """
    reached goal / entered the goal arm but never touched the goal / got into the connecting corridor
    cell (2,3) but never into the goal arm / never reached the corridor (stayed on the start side of the wall).
    """
    if success_step is not None:
        return OUTCOMES[0]
    goal_row = 4 - start_cell[0]     # the opposite arm
    cells = {xy_to_cell(x, y) for x, y in traj}
    if any(r == goal_row for r, _ in cells):
        return OUTCOMES[1]
    if (2, 3) in cells:
        return OUTCOMES[2]
    return OUTCOMES[3]


def run_episode(model, env, seed, start_cell, goal_cell, device, render=False, header="", sub=()):
    obs, _ = env.reset(seed=seed, options={"reset_cell": np.array(start_cell), "goal_cell": np.array(goal_cell)})
    follow = setup_camera(env, ENV_ID) if render else None
    goals, states, masks = model.init_obj()
    hd = model.worker.wrnn.hidden_size
    model.worker_hidden = (torch.zeros(1, hd, device=device), torch.zeros(1, hd, device=device))
    model.manager_hidden = model.manager.mrnn.init_hidden(1, device)
    goal_xy = obs[2:4].copy()        # flattened Dict obs: achieved_goal(2), desired_goal(2), observation(27)
    traj, total, success_step, frames, t, done = [obs[:2].copy()], 0.0, None, [], 0, False
    while not done:
        with torch.no_grad():
            dist, goals, states, _, _ = model(np.expand_dims(obs, 0), goals, states, masks[-1])
        action = np.clip(dist.mean.cpu().numpy()[0], -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        t += 1
        traj.append(obs[:2].copy())
        done = terminated or truncated
        masks.pop(0)
        masks.append(torch.tensor([[0.0 if done else 1.0]], device=device))
        if info.get("success", False) and success_step is None:
            success_step = t
        if render and t % FRAME_SKIP == 0:
            follow()
            d = float(np.linalg.norm(obs[:2] - goal_xy))
            frames.append(annotate(env.render(), [header, *sub,
                                                  f"step {t}   distance to goal {d:4.1f} m   return {total:,.1f}"
                                                  + ("   GOAL REACHED" if success_step else "")]))
        if render and success_step is not None and t >= success_step + AFTER_SUCCESS:
            break
    return dict(traj=np.array(traj), goal=goal_xy, ret=total, success_step=success_step, length=t, frames=frames)


def annotate(frame, lines):
    """Caption in a black bar above the frame, so it never covers the maze."""
    bar = 6 + 19 * len(lines)
    img = Image.new("RGB", (frame.shape[1], frame.shape[0] + bar), (0, 0, 0))
    img.paste(Image.fromarray(frame), (0, bar))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("segoeui.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for i, s in enumerate(lines):
        draw.text((6, 4 + 19 * i), s, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def episode_stats(ep, start_cell):
    traj, goal = ep["traj"], ep["goal"]
    d = np.linalg.norm(traj - goal, axis=1)
    cells = [xy_to_cell(x, y) for x, y in traj]
    # Share of the episode spent in the start arm's two cells that are walled off from the goal
    behind_wall = np.mean([c[0] == start_cell[0] and c[1] in (1, 2) for c in cells])
    path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    return dict(min_d=d.min(), final_d=d[-1], behind_wall_frac=behind_wall, path_len=path_len,
                final_cell=cells[-1])


# Reference palette, light mode
SURFACE, INK, INK_2, MUTED, WALL = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#d6d5cd"
OUT_COLOR = {OUTCOMES[0]: "#2a78d6", OUTCOMES[1]: "#eb6834", OUTCOMES[2]: "#1baf7a", OUTCOMES[3]: "#898781"}


def plot(results, n_eps):
    fig, axes = plt.subplots(len(MODELS), len(PAIRS), figsize=(3.3 * len(PAIRS), 3.6 * len(MODELS) + 0.8),
                             facecolor=SURFACE)
    for r, (model_name, _, model_type) in enumerate(MODELS):
        for c, (pair_name, start_cell, goal_cell, _) in enumerate(PAIRS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            for i in range(5):
                for j in range(5):
                    if MAZE[i][j]:
                        x0, y0 = j * SCALE - CENTER, CENTER - (i + 1) * SCALE
                        ax.add_patch(plt.Rectangle((x0, y0), SCALE, SCALE, color=WALL, linewidth=0))
            eps = results[(model_name, pair_name)]
            for ep in sorted(eps, key=lambda e: OUTCOMES.index(e["outcome"]), reverse=True):
                tr = ep["traj"][: (ep["success_step"] or ep["length"]) + 1]
                ax.plot(tr[:, 0], tr[:, 1], color=OUT_COLOR[ep["outcome"]], linewidth=1.1, alpha=0.85)
                ax.plot(*tr[0], marker="o", markersize=4, markerfacecolor=SURFACE,
                        markeredgecolor=OUT_COLOR[ep["outcome"]], markeredgewidth=1.2)
                ax.plot(*ep["goal"], marker="*", markersize=9, color=INK, markeredgecolor=SURFACE,
                        markeredgewidth=0.6)
            counts = Counter(e["outcome"] for e in eps)
            ax.set_title(f"{MODEL_NAME[model_type]} · start {start_cell} → goal {goal_cell}", fontsize=9.5,
                         color=INK, loc="left", pad=16)
            ax.text(0, 1.015, f"goal {counts[OUTCOMES[0]]}/{n_eps} · missed {counts[OUTCOMES[1]]} · "
                              f"turned back {counts[OUTCOMES[2]]} · stuck {counts[OUTCOMES[3]]}",
                    transform=ax.transAxes, fontsize=7.5, color=INK_2, va="bottom")
            ax.set_xlim(-CENTER, CENTER)
            ax.set_ylim(-CENTER, CENTER)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    handles = [plt.Line2D([], [], color=OUT_COLOR[o], linewidth=2) for o in OUTCOMES]
    handles += [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_2),
                plt.Line2D([], [], marker="*", linestyle="", color=INK, markersize=9)]
    fig.legend(handles, OUTCOMES + ["start", "goal"], loc="upper left", ncol=6, frameon=False, fontsize=8.5,
               labelcolor=INK_2, bbox_to_anchor=(0.005, 0.955))
    fig.suptitle("AntMaze — goal behind the wall: does the ant go around?", x=0.01, y=0.99, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.01, f"{n_eps} episodes per panel, deterministic actions, ±1 m start/goal noise. Gray squares "
                         "are walls; the only way between the two arms is the right corridor. Lines stop at the "
                         "first goal touch.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93), h_pad=2.0)
    fig.savefig(PLOT, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", PLOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(args.device)
    results = {}
    for model_name, rel, model_type in MODELS:
        env = make_env(ENV_ID, render=False)
        model, ck_step, seed = load_model(os.path.join(CK, rel), model_type, {"spatial_obs_norm": "running"},
                                          env.observation_space.shape[0], env.action_space.shape[0], device)
        seeds = [seed + 2000 + i for i in range(args.episodes)]
        for pair_name, start_cell, goal_cell, desc in PAIRS:
            eps = []
            for s in seeds:
                ep = run_episode(model, env, s, start_cell, goal_cell, device)
                ep["seed"] = s
                ep["outcome"] = classify(ep["traj"], start_cell, ep["success_step"])
                ep.update(episode_stats(ep, start_cell))
                eps.append(ep)
            results[(model_name, pair_name)] = eps
            counts = Counter(e["outcome"] for e in eps)
            print(f"\n== {MODEL_NAME[model_type]} @ {ck_step/1e6:.2f}M  {pair_name} {start_cell}->{goal_cell}: "
                  + ", ".join(f"{o} {counts[o]}" for o in OUTCOMES))
            for e in eps:
                print(f"  seed {e['seed']}: {e['outcome']:<26} success@{str(e['success_step']):>4}  "
                      f"min d {e['min_d']:4.1f} m  final d {e['final_d']:4.1f} m  "
                      f"behind wall {e['behind_wall_frac']*100:3.0f}% of steps  walked {e['path_len']:5.1f} m  "
                      f"final cell {e['final_cell']}  return {e['ret']:6.1f}")
            if args.no_gif:
                continue
            # GIF 1: the most common outcome (first seed with it). GIF 2: the furthest-getting outcome, if different
            mode = counts.most_common(1)[0][0]
            picks = [(mode, next(e for e in eps if e["outcome"] == mode))]
            best = min(counts, key=OUTCOMES.index)
            if best != mode:
                cand = [e for e in eps if e["outcome"] == best]
                ep = min(cand, key=lambda e: e["success_step"]) if best == OUTCOMES[0] else \
                    min(cand, key=lambda e: e["min_d"])
                picks.append((best, ep))
            render_env = make_env(ENV_ID, render=True)
            for outcome, ep in picks:
                header = f"{MODEL_NAME[model_type]} · checkpoint @ {ck_step/1e6:.2f}M steps · seed {ep['seed']}"
                rec = run_episode(model, render_env, ep["seed"], start_cell, goal_cell, device, render=True,
                                  header=header,
                                  sub=(desc, f"this episode: {outcome} ({counts[outcome]}/{len(eps)} episodes)"))
                out = os.path.join(OUT_DIR, f"{model_name}_{pair_name}_{TAGS[outcome]}.gif")
                imageio.mimsave(out, rec["frames"], duration=FRAME_MS, loop=0)
                same = "" if (rec["success_step"] == ep["success_step"]) else "  (WARNING: re-run differs)"
                print(f"  gif: seed {ep['seed']} {outcome}, {rec['length']} steps -> {out} "
                      f"({os.path.getsize(out)/1e6:.1f} MB){same}")
            render_env.close()
        env.close()
    plot(results, args.episodes)


if __name__ == "__main__":
    main()
