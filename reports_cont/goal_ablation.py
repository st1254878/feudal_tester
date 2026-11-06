"""
Goal ablation on the final HalfCheetah-v4 / Ant-v4 checkpoints (V4 code): does the Worker need the Manager's goals?

The Manager still runs every step; only the goal it hands to the Worker is replaced:
  real      - unchanged
  frozen    - the Manager's first goal of the episode, repeated every step
  random10  - a random unit vector, redrawn every 10 steps
  random1   - a random unit vector, redrawn every step
  zero      - all zeros
Deterministic (mean) actions, the same 32 env seeds for every condition, and the same random-goal draws per seed.
Writes reports_cont/goal_ablation/<key>.json.

Usage: python reports_cont/goal_ablation.py --device cuda [--only cheetah_fun] [--episodes 32]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_peaks import CK, load_model, make_env  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal_ablation")
RUNS = {
    "cheetah_fun": ("HalfCheetah-v4", "20261018_021526_cont_cheetah_fun_10m/model_final_10000000.pt", "fun"),
    "cheetah_cat": ("HalfCheetah-v4", "20261018_021543_cont_cheetah_cat_10m/model_final_10000000.pt", "feudal_cat"),
    "ant_fun": ("Ant-v4", "20261018_114728_cont_ant_fun_20m/model_final_20000000.pt", "fun"),
    "ant_cat": ("Ant-v4", "20261018_114744_cont_ant_cat_20m/model_final_20000000.pt", "feudal_cat"),
}
CONDITIONS = ["real", "frozen", "random10", "random1", "zero"]


def install_goal_ablation(model):
    """Wrap the Manager so the goal it returns can be replaced; returns a controller dict."""
    ctl = {"mode": "real", "t": 0, "first": None, "rand": None, "gen": None}
    orig = model.manager.forward

    def forward(*args, **kwargs):
        goal, hidden, state, value = orig(*args, **kwargs)
        mode, t = ctl["mode"], ctl["t"]
        if ctl["first"] is None:
            ctl["first"] = goal.detach().clone()
        if mode == "frozen":
            goal = ctl["first"]
        elif mode in ("random10", "random1"):
            every = 10 if mode == "random10" else 1
            if t % every == 0:
                noise = torch.randn(goal.shape, generator=ctl["gen"], device=goal.device)
                ctl["rand"] = F.normalize(noise, dim=-1)
            goal = ctl["rand"]
        elif mode == "zero":
            goal = torch.zeros_like(goal)
        ctl["t"] += 1
        return goal, hidden, state, value

    model.manager.forward = forward
    return ctl


def run_episode(model, env, ctl, mode, seed, device):
    ctl.update(mode=mode, t=0, first=None, rand=None)
    ctl["gen"] = torch.Generator(device=device).manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    goals, states, masks = model.init_obj()
    hd = model.worker.wrnn.hidden_size
    model.worker_hidden = (torch.zeros(1, hd, device=device), torch.zeros(1, hd, device=device))
    model.manager_hidden = model.manager.mrnn.init_hidden(1, device)
    total, t, done = 0.0, 0, False
    x0 = float(env.unwrapped.data.qpos[0])
    with torch.no_grad():
        while not done:
            dist, goals, states, _, _ = model(np.expand_dims(obs, 0), goals, states, masks[-1])
            obs, reward, terminated, truncated, _ = env.step(np.clip(dist.mean.cpu().numpy()[0], -1.0, 1.0))
            total += reward
            t += 1
            done = terminated or truncated
            masks.pop(0)
            masks.append(torch.tensor([[0.0 if done else 1.0]], device=device))
    return total, t, float(env.unwrapped.data.qpos[0]) - x0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", default=None)
    p.add_argument("--episodes", type=int, default=32)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    for key, (env_id, rel, model_type) in RUNS.items():
        if args.only and args.only != key:
            continue
        env = make_env(env_id, render=False)
        model, step, seed = load_model(os.path.join(CK, rel), model_type, {}, env.observation_space.shape[0],
                                       env.action_space.shape[0], device)
        ctl = install_goal_ablation(model)
        seeds = [seed + 1000 + i for i in range(args.episodes)]
        result = {"env": env_id, "checkpoint": rel, "model_type": model_type, "step": step, "seeds": seeds,
                  "time_horizon": model.worker.c, "conditions": {}}
        for mode in CONDITIONS:
            eps = [run_episode(model, env, ctl, mode, s, device) for s in seeds]
            ret = np.array([e[0] for e in eps])
            result["conditions"][mode] = {"returns": ret.tolist(), "lengths": [e[1] for e in eps],
                                          "distance": [e[2] for e in eps]}
            print(f"{key:<12} {mode:<9} mean {ret.mean():8.1f}  median {np.median(ret):8.1f}  "
                  f"min {ret.min():8.1f}  max {ret.max():8.1f}  mean length {np.mean([e[1] for e in eps]):6.0f}",
                  flush=True)
        env.close()
        with open(os.path.join(OUT, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
