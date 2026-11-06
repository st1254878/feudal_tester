"""
eval_all_checkpoints.py - Evaluates all intermediate checkpoints using the new fixed-seed
deterministic evaluation mechanism (20 fixed seeds across parallel vectorized environments)
and generates high-quality smoothed comparison plots and CSV reports.

Usage:
    # Run full evaluation across all checkpoints of both FuN and FeUdalCat:
    python eval_all_checkpoints.py

    # Or specify custom directories:
    python eval_all_checkpoints.py --fun-dir checkpoints_continuous/20261017_105223_cont_cheetah_fun_fixed --cat-dir checkpoints_continuous/20261017_105246_cont_cheetah_cat_fixed
"""

import os
import re
import csv
import argparse
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from feudalnet_continuous import ContinuousFeUdalNetwork
from feudal_cat_continuous import ContinuousFeUdalCatNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints using fixed-seed evaluation")
    parser.add_argument("--fun-dir", type=str, default="checkpoints_continuous/20261017_105223_cont_cheetah_fun_fixed",
                        help="Path to Standard FuN checkpoints directory")
    parser.add_argument("--cat-dir", type=str, default="checkpoints_continuous/20261017_105246_cont_cheetah_cat_fixed",
                        help="Path to FeUdalCat checkpoints directory")
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v4",
                        help="Gymnasium environment ID (default: HalfCheetah-v4)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of fixed-seed evaluation episodes (default: 20)")
    parser.add_argument("--seed", type=int, default=1042,
                        help="Base evaluation seed (default: 1042)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Step stride (1 = evaluate all, 2 = evaluate every 2nd checkpoint)")
    parser.add_argument("--out-plot", type=str, default="plots/cheetah_fist_half_comparison.png",
                        help="Output image path for comparison chart")
    parser.add_argument("--out-csv", type=str, default="plots/cheetah_first_half_comparison.csv",
                        help="Output CSV path for evaluation results")
    return parser.parse_args()


def gaussian_smooth(y, sigma=2.0):
    y = np.asarray(y, dtype=float)
    if len(y) <= 3:
        return y
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(y, radius, mode="edge")
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def get_checkpoint_files(ckpt_dir):
    """Finds and sorts all step checkpoints in chronological step order."""
    if not os.path.exists(ckpt_dir):
        return []

    ckpts = []
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".pt"):
            continue
        full_path = os.path.join(ckpt_dir, fname)

        # Match model_step_*.pt or model_final_*.pt
        match = re.search(r"model_(?:step|final)_(\d+)\.pt", fname)
        if match:
            step = int(match.group(1))
            ckpts.append((step, full_path))

    # Sort strictly by environment step count
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def make_vec_eval_envs(env_id, num_envs, base_seed):
    """Creates vectorized environments with pre-assigned fixed seeds."""
    def make_single(idx):
        def _thunk():
            env = gym.make(env_id)
            env.action_space.seed(base_seed + idx)
            return env
        return _thunk

    return gym.vector.SyncVectorEnv([make_single(i) for i in range(num_envs)])


def evaluate_checkpoint_list(model_cls, ckpt_list, envs, num_envs, base_seed, model_name, device):
    """Evaluates all checkpoints in parallel across vectorized environments."""
    if not ckpt_list:
        print(f"[-] No checkpoints found for {model_name}!")
        return [], [], []

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    # Inspect first checkpoint to extract exact architecture hyperparameters
    first_ckpt = torch.load(ckpt_list[0][1], map_location=device, weights_only=False)
    ckpt_args = first_ckpt.get("args", {})
    d = ckpt_args.get("hidden_dim_manager", 256)
    k = ckpt_args.get("hidden_dim_worker", 16)
    c = ckpt_args.get("time_horizon", 10)
    r = ckpt_args.get("dilation", 10)

    model = model_cls(
        num_workers=num_envs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim_manager=d,
        hidden_dim_worker=k,
        time_horizon=c,
        dilation=r,
        device=device
    ).to(device)
    model.eval()

    steps = []
    means = []
    stds = []

    total_ckpts = len(ckpt_list)
    print(f"\n[*] Evaluating {total_ckpts} checkpoints for [{model_name}] using {num_envs} fixed seeds (seeds {base_seed}..{base_seed+num_envs-1})...")

    seeds_list = [base_seed + i for i in range(num_envs)]

    for idx, (step, ckpt_path) in enumerate(ckpt_list, 1):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

        # Reset recurrent memory cleanly
        goals, states, masks = model.init_obj()
        worker_hidden_dim = model.worker.wrnn.hidden_size
        model.worker_hidden = (
            torch.zeros(num_envs, worker_hidden_dim, device=device),
            torch.zeros(num_envs, worker_hidden_dim, device=device)
        )
        model.manager_hidden = model.manager.mrnn.init_hidden(num_envs, device)

        # Reset environments with fixed seeds
        obs, _ = envs.reset(seed=seeds_list)
        ep_returns = np.zeros(num_envs, dtype=float)

        with torch.no_grad():
            for t in range(1000):
                dist, goals, states, _, _ = model(obs, goals, states, masks[-1])
                # Deterministic evaluation using distribution mean
                action = np.clip(dist.mean.cpu().numpy(), -1.0, 1.0)
                obs, reward, term, trunc, _ = envs.step(action)
                ep_returns += reward

                masks.pop(0)
                masks.append(torch.ones(num_envs, 1, device=device))

        mean_ret = float(np.mean(ep_returns))
        std_ret = float(np.std(ep_returns))
        steps.append(step)
        means.append(mean_ret)
        stds.append(std_ret)

        print(f"  [{model_name} {idx:2d}/{total_ckpts:2d}] Step {step:9,d} | Eval Return: {mean_ret:7.1f} +/- {std_ret:5.1f} (Min: {np.min(ep_returns):6.1f}, Max: {np.max(ep_returns):6.1f})")

    return np.array(steps), np.array(means), np.array(stds)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"  Batch Checkpoint Evaluation (Fixed-Seed Mechanism)")
    print(f"  Environment : {args.env_name}")
    print(f"  Device      : {device}")
    print(f"  Fixed Seeds : {args.episodes} episodes (seeds {args.seed}..{args.seed+args.episodes-1})")
    print(f"  FuN Dir     : {args.fun_dir}")
    print(f"  Cat Dir     : {args.cat_dir}")
    print(f"==================================================")

    # 1. Collect and filter checkpoints
    fun_ckpts = get_checkpoint_files(args.fun_dir)[::args.stride]
    cat_ckpts = get_checkpoint_files(args.cat_dir)[::args.stride]

    if not fun_ckpts and not cat_ckpts:
        print("[-] Error: No checkpoint files found. Please verify the directory paths.")
        return

    # 2. Build vectorized environment (reused across all checkpoints for high throughput)
    print(f"[*] Initializing {args.episodes} parallel simulation environments on CPU/GPU...")
    envs = make_vec_eval_envs(args.env_name, args.episodes, args.seed)

    # 3. Evaluate FuN checkpoints
    fun_steps, fun_means, fun_stds = evaluate_checkpoint_list(
        ContinuousFeUdalNetwork, fun_ckpts, envs, args.episodes, args.seed, "Standard FuN", device
    )

    # 4. Evaluate FeUdalCat checkpoints
    cat_steps, cat_means, cat_stds = evaluate_checkpoint_list(
        ContinuousFeUdalCatNetwork, cat_ckpts, envs, args.episodes, args.seed, "FeUdalCat", device
    )

    envs.close()

    # 5. Export to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", "step", "eval_mean_fixed_seed", "eval_std"])
        for s, m, sd in zip(fun_steps, fun_means, fun_stds):
            writer.writerow(["Standard FuN", int(s), float(m), float(sd)])
        for s, m, sd in zip(cat_steps, cat_means, cat_stds):
            writer.writerow(["FeUdalCat", int(s), float(m), float(sd)])
    print(f"\n[+] Exported evaluation results to CSV: {args.out_csv}")

    # 6. Generate Comparison Plot
    os.makedirs(os.path.dirname(os.path.abspath(args.out_plot)), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    colors = {"Standard FuN": "#1f77b4", "FeUdalCat": "#ff7f0e"}

    for name, s, v, sd in [("Standard FuN", fun_steps, fun_means, fun_stds), ("FeUdalCat", cat_steps, cat_means, cat_stds)]:
        if len(s) == 0:
            continue
        sm = gaussian_smooth(v, sigma=2.0)
        c = colors[name]

        # Raw evaluation points
        ax.plot(s, v, color=c, alpha=0.25, linestyle="--", linewidth=1.2, label=f"{name} (Raw Fixed Seed)")
        # Smoothed curve
        lbl = f"{name} (Smoothed | Best Raw: {np.max(v):.0f}, Best Smooth: {np.max(sm):.0f})"
        ax.plot(s, sm, color=c, linewidth=2.8, label=lbl)
        # Translucent confidence band
        sm_std = gaussian_smooth(sd, sigma=2.0)
        ax.fill_between(s, sm - 0.5 * sm_std, sm + 0.5 * sm_std, color=c, alpha=0.10)

    ax.set_title(f"HalfCheetah-v4 Checkpoint Evaluation (New Mechanism: Fixed Seeds 1042..{args.seed+args.episodes-1}, {args.episodes} Episodes)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Deterministic Evaluation Reward", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    print(f"[+] Saved high-resolution comparison chart to: {args.out_plot}")

    # 7. Print Milestone Summary Table
    print("\n" + "=" * 65)
    print(f"  Milestone Summary Table (Fixed {args.episodes} Seeds: {args.seed}..{args.seed+args.episodes-1})")
    print("=" * 65)
    print("  Step   | Standard FuN (Raw / Smooth)   | FeUdalCat (Raw / Smooth)")
    print("-" * 65)
    milestones = [100000, 500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000]

    fun_sm = gaussian_smooth(fun_means, sigma=2.0) if len(fun_means) > 0 else np.array([])
    cat_sm = gaussian_smooth(cat_means, sigma=2.0) if len(cat_means) > 0 else np.array([])

    for m in milestones:
        fun_str = "    N/A /     N/A"
        cat_str = "    N/A /     N/A"
        if len(fun_steps) > 0:
            i_f = np.argmin(np.abs(fun_steps - m))
            fun_str = f"{fun_means[i_f]:7.1f} / {fun_sm[i_f]:7.1f}"
        if len(cat_steps) > 0:
            i_c = np.argmin(np.abs(cat_steps - m))
            cat_str = f"{cat_means[i_c]:7.1f} / {cat_sm[i_c]:7.1f}"
        print(f"  {m//1000:4d}k  | {fun_str:29s} | {cat_str:24s}")
    print("=" * 65)


if __name__ == "__main__":
    main()
