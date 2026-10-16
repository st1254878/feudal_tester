"""
plot_results.py - Visualizes training metrics from TensorBoard event files in runs/
Supports plotting reward curves, manager/worker metrics, and exporting to CSV.

Usage:
    # List all runs:
    python plot_results.py --list

    # Plot a specific run:
    python plot_results.py --run runs/20261015_163553_cont_cheetah_fun

    # Plot latest run:
    python plot_results.py --latest

    # Export CSV as well:
    python plot_results.py --latest --csv
"""

import os
import csv
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def get_all_runs(runs_dirs=("runs", "runs_continuous")):
    all_runs = []
    for rdir in runs_dirs:
        if os.path.exists(rdir):
            for d in os.listdir(rdir):
                full = os.path.join(rdir, d)
                if os.path.isdir(full):
                    all_runs.append(full)
    all_runs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return all_runs


def load_scalars(run_path):
    print(f"[*] Loading TensorBoard events from: {run_path}")
    ea = event_accumulator.EventAccumulator(
        run_path,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0 means load all events without downsampling
        }
    )
    ea.Reload()
    available_tags = ea.Tags().get("scalars", [])
    
    data = {}
    for tag in available_tags:
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": np.array([e.step for e in events]),
            "values": np.array([e.value for e in events]),
            "wall_times": np.array([e.wall_time for e in events]),
        }
    return data


def gaussian_smooth(y, sigma=1.5):
    y = np.asarray(y, dtype=float)
    if len(y) <= 3:
        return y
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(y, radius, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[radius:-radius]


def moving_average(values, window=50):
    if len(values) == 0:
        return np.array([])
    if len(values) < window:
        window = max(1, len(values) // 2)
    if window <= 1:
        return np.array(values)
    # Cumulative sum moving average
    cumsum = np.cumsum(np.insert(values, 0, 0))
    res = np.empty(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        res[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)
    return res


def plot_metrics(run_path, data, out_path=None, window=50, export_csv=False):
    run_name = os.path.basename(os.path.normpath(run_path))
    
    if out_path is None:
        os.makedirs("plots", exist_ok=True)
        out_path = os.path.join("plots", f"{run_name}.png")
    else:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.suptitle(f"Training Progress: {run_name}", fontsize=14, fontweight="bold")

    # 1. Episode Reward
    ax1 = axes[0, 0]
    has_reward = False

    if "eval/mean_reward" in data:
        # Plot purely the smoothed evaluation curve with translucent distribution band (no scatter points)
        eval_steps = data["eval/mean_reward"]["steps"]
        eval_means = data["eval/mean_reward"]["values"]
        eval_stds = data["eval/std_reward"]["values"] if "eval/std_reward" in data and len(data["eval/std_reward"]["values"]) == len(eval_means) else np.zeros_like(eval_means)

        sigma = 1.8 if len(eval_means) > 10 else 1.0
        smoothed_mean = gaussian_smooth(eval_means, sigma=sigma)
        smoothed_std = gaussian_smooth(eval_stds, sigma=sigma)

        # Translucent background band for distribution variance
        ax1.fill_between(
            eval_steps,
            smoothed_mean - smoothed_std,
            smoothed_mean + smoothed_std,
            color="royalblue",
            alpha=0.18,
            label=r"Score Variance ($\pm 1\,\sigma$)"
        )

        # Clean, smooth curve without individual point markers
        ax1.plot(
            eval_steps,
            smoothed_mean,
            color="royalblue",
            linewidth=2.5,
            label="Evaluation Return (Smoothed)"
        )

        ax1.set_title("Deterministic Evaluation Return", fontweight="bold")
        ax1.set_xlabel("Environment Steps")
        ax1.set_ylabel("Reward")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper left")

        max_eval = np.max(eval_means) if len(eval_means) > 0 else 0.0
        last_eval = eval_means[-1] if len(eval_means) > 0 else 0.0
        ax1.text(0.97, 0.05, f"Best Eval: {max_eval:.1f}\nLatest Eval: {last_eval:.1f}",
                 transform=ax1.transAxes, verticalalignment="bottom", horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"))
        has_reward = True

    elif "episode/reward" in data:
        # Fallback for training runs without eval (e.g. Atari Montezuma)
        steps = data["episode/reward"]["steps"]
        rewards = data["episode/reward"]["values"]
        smoothed = moving_average(rewards, window=window)
        ax1.plot(steps, rewards, alpha=0.20, color="dodgerblue", label="Raw Train Reward")
        ax1.plot(steps, smoothed, color="blue", linewidth=1.5, label=f"Train Moving Avg (w={window})")
        max_train_r = np.max(rewards) if len(rewards) > 0 else 0.0
        last_100_mean = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        ax1.set_title("Episode Reward", fontweight="bold")
        ax1.set_xlabel("Environment Steps")
        ax1.set_ylabel("Reward")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper left")
        ax1.text(0.97, 0.05, f"Max: {max_train_r:.1f}\nLast 100: {last_100_mean:.1f}",
                 transform=ax1.transAxes, verticalalignment="bottom", horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"))
        has_reward = True

    if not has_reward:
        ax1.text(0.5, 0.5, "No reward metrics found", ha="center", va="center")

    # 2. Episode Length
    ax2 = axes[0, 1]
    if "eval/mean_length" in data:
        eval_steps = data["eval/mean_length"]["steps"]
        eval_lens = data["eval/mean_length"]["values"]
        smoothed_lens = gaussian_smooth(eval_lens, sigma=1.5)
        ax2.plot(eval_steps, smoothed_lens, color="forestgreen", linewidth=2.0, label="Eval Episode Length")
        ax2.set_title("Evaluation Episode Length", fontweight="bold")
        ax2.set_xlabel("Environment Steps")
        ax2.set_ylabel("Steps")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left")
    elif "episode/length" in data:
        steps = data["episode/length"]["steps"]
        lengths = data["episode/length"]["values"]
        smoothed_len = moving_average(lengths, window=window)
        ax2.plot(steps, lengths, alpha=0.25, color="forestgreen", label="Raw Length")
        ax2.plot(steps, smoothed_len, color="darkgreen", linewidth=1.8, label=f"Moving Avg (w={window})")
        ax2.set_title("Episode Length (Steps Survived)", fontweight="bold")
        ax2.set_xlabel("Environment Steps")
        ax2.set_ylabel("Steps")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left")
    else:
        ax2.text(0.5, 0.5, "No 'episode/length' found", ha="center", va="center")

    # 3. FuN Specific Metrics (Manager Cosine & Worker Intrinsic Reward)
    ax3 = axes[1, 0]
    has_fun = False
    if "policy/manager_cosines" in data:
        steps = data["policy/manager_cosines"]["steps"]
        cosines = data["policy/manager_cosines"]["values"]
        ax3.plot(steps, cosines, color="purple", label="Manager Cosine Sim", linewidth=1.2)
        has_fun = True
    if "policy/worker_intrinsic_reward" in data:
        steps = data["policy/worker_intrinsic_reward"]["steps"]
        ir = data["policy/worker_intrinsic_reward"]["values"]
        ax3.plot(steps, ir, color="darkorange", label="Worker Intrinsic Reward", linewidth=1.2)
        has_fun = True
    
    if has_fun:
        ax3.set_title("FuN Internal Dynamics (Manager & Worker)", fontweight="bold")
        ax3.set_xlabel("Environment Steps")
        ax3.set_ylabel("Value")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="upper left")
    else:
        ax3.text(0.5, 0.5, "No internal FuN policy metrics recorded", ha="center", va="center")

    # 4. Losses
    ax4 = axes[1, 1]
    has_loss = False
    if "loss/total" in data:
        ax4.plot(data["loss/total"]["steps"], data["loss/total"]["values"], label="Total Loss", color="crimson", alpha=0.8)
        has_loss = True
    if "loss/manager" in data:
        ax4.plot(data["loss/manager"]["steps"], data["loss/manager"]["values"], label="Manager Loss", color="indigo", alpha=0.6)
        has_loss = True
    if "loss/worker" in data:
        ax4.plot(data["loss/worker"]["steps"], data["loss/worker"]["values"], label="Worker Loss", color="teal", alpha=0.6)
        has_loss = True

    if has_loss:
        ax4.set_title("Training Losses", fontweight="bold")
        ax4.set_xlabel("Environment Steps")
        ax4.set_ylabel("Loss")
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(loc="upper right")
    else:
        ax4.text(0.5, 0.5, "No loss metrics recorded", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved visualization plot to: {out_path}")

    # Export CSV if requested
    if export_csv:
        csv_path = out_path.replace(".png", ".csv")
        if "eval/mean_reward" in data:
            eval_steps = data["eval/mean_reward"]["steps"]
            eval_means = data["eval/mean_reward"]["values"]
            eval_stds = data["eval/std_reward"]["values"] if "eval/std_reward" in data and len(data["eval/std_reward"]["values"]) == len(eval_means) else np.zeros_like(eval_means)
            sigma = 1.8 if len(eval_means) > 10 else 1.0
            smoothed_mean = gaussian_smooth(eval_means, sigma=sigma)
            smoothed_std = gaussian_smooth(eval_stds, sigma=sigma)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "eval_mean_raw", "eval_mean_smoothed", "eval_std_smoothed"])
                for i in range(len(eval_steps)):
                    writer.writerow([int(eval_steps[i]), float(eval_means[i]), float(smoothed_mean[i]), float(smoothed_std[i])])
            print(f"[+] Exported Eval CSV data to: {csv_path}")

        elif "episode/reward" in data:
            steps = data["episode/reward"]["steps"]
            rewards = data["episode/reward"]["values"]
            smoothed = moving_average(rewards, window=window)
            lengths = data["episode/length"]["values"] if ("episode/length" in data and len(data["episode/length"]["values"]) == len(steps)) else None
            
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["step", "reward", f"reward_smoothed_w{window}"]
                if lengths is not None:
                    header.append("length")
                writer.writerow(header)
                
                for i in range(len(steps)):
                    row = [int(steps[i]), float(rewards[i]), float(smoothed[i])]
                    if lengths is not None:
                        row.append(int(lengths[i]))
                    writer.writerow(row)
            print(f"[+] Exported CSV data to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot FeUdal Networks training results from TensorBoard logs")
    parser.add_argument("--run", type=str, default=None, help="Path to run directory (e.g. runs/20261015_163553_cont_cheetah_fun)")
    parser.add_argument("--latest", action="store_true", help="Plot the most recent run")
    parser.add_argument("--list", action="store_true", help="List all available runs in runs/")
    parser.add_argument("--window", type=int, default=50, help="Moving average window size (default: 50)")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output image file path (default: plots/<run_name>.png)")
    parser.add_argument("--csv", action="store_true", help="Also export smoothed rewards to CSV")
    args = parser.parse_args()

    all_runs = get_all_runs()

    if args.list:
        print("\n=== Available Runs (runs/ & runs_continuous/) ===")
        for i, r in enumerate(all_runs):
            print(f"[{i+1}] {r}")
        print()
        return

    target_run = None
    if args.latest:
        if not all_runs:
            print("[-] No runs found in 'runs/' folder.")
            return
        target_run = all_runs[0]
    elif args.run:
        target_run = args.run
    else:
        # Default behavior: list and ask or pick latest
        if not all_runs:
            print("[-] No runs found in 'runs/' folder.")
            return
        target_run = all_runs[0]
        print(f"[*] No run specified, picking latest: {target_run}")

    if not os.path.exists(target_run):
        print(f"[-] Run directory does not exist: {target_run}")
        return

    data = load_scalars(target_run)
    plot_metrics(target_run, data, out_path=args.out, window=args.window, export_csv=args.csv)


if __name__ == "__main__":
    main()
