#!/usr/bin/env python3
"""
=============================================================================
AntMaze 雙軌對比繪圖腳本 (FuN vs FeUdalCat Comparison Plotter)
=============================================================================
功能：
  自動搜尋 runs_continuous/ 下最新的 FuN 與 FeUdalCat 記錄，
  在同一張圖表上對比展示：
  1. 確定性評估回報 (Deterministic Return) 曲線
  2. 迷宮成功率 (Success Rate %) 曲線
  並在終端輸出即時對比摘要表格。

使用方式：
  python compare_runs.py
  python compare_runs.py --fun_dir runs_continuous/<fun_dir> --cat_dir runs_continuous/<cat_dir>
=============================================================================
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 設定 Windows 終端編碼
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 設定繁體中文字型與負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_run_matching(pattern, root="runs_continuous"):
    candidates = []
    for d in glob.glob(os.path.join(root, "*")):
        if os.path.isdir(d) and pattern in os.path.basename(d).lower():
            tfevents = [f for f in os.listdir(d) if f.startswith("events.out.tfevents")]
            if tfevents:
                candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def load_eval_scalars(run_dir):
    if not run_dir or not os.path.isdir(run_dir):
        return {}
    ef_list = [f for f in os.listdir(run_dir) if f.startswith("events.out.tfevents")]
    if not ef_list:
        return {}
    ef_path = os.path.join(run_dir, ef_list[0])
    ea = event_accumulator.EventAccumulator(ef_path)
    ea.Reload()

    scalar_data = {}
    available_tags = ea.Tags().get("scalars", [])
    target_tags = ["eval/mean_reward", "eval/success_rate", "loss/total"]
    for tag in target_tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            scalar_data[tag] = (np.array(steps), np.array(vals))
    return scalar_data


def smooth_curve(vals, weight=0.6):
    if len(vals) == 0:
        return vals
    smoothed = []
    last = vals[0]
    for v in vals:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def main():
    parser = argparse.ArgumentParser(description="Compare FuN vs FeUdalCat runs")
    parser.add_argument("--fun_dir", type=str, default="", help="Path to FuN run directory")
    parser.add_argument("--cat_dir", type=str, default="", help="Path to FeUdalCat run directory")
    parser.add_argument("--output", type=str, default="plots/fun_vs_feudal_cat.png", help="Output plot image path")
    args = parser.parse_args()

    fun_dir = args.fun_dir or find_latest_run_matching("fun")
    cat_dir = args.cat_dir or find_latest_run_matching("cat")

    print("=" * 70)
    print("📊 AntMaze 模型性能對比 (FuN vs FeUdalCat)")
    print("=" * 70)
    print(f"Standard FuN 目錄:      {fun_dir or '[未找到]'}")
    print(f"FeUdalCat 目錄:         {cat_dir or '[未找到]'}")
    print("=" * 70)

    if not fun_dir and not cat_dir:
        print("[!] 錯誤: 未在 runs_continuous/ 找到任何 FuN 或 FeUdalCat 訓練記錄。")
        return

    data_fun = load_eval_scalars(fun_dir) if fun_dir else {}
    data_cat = load_eval_scalars(cat_dir) if cat_dir else {}

    # Print Summary Table
    def get_stats(data):
        ret_steps, ret_vals = data.get("eval/mean_reward", ([], []))
        succ_steps, succ_vals = data.get("eval/success_rate", ([], []))
        latest_step = ret_steps[-1] if len(ret_steps) > 0 else 0
        latest_ret = ret_vals[-1] if len(ret_vals) > 0 else 0.0
        best_ret = np.max(ret_vals) if len(ret_vals) > 0 else 0.0
        peak_succ = np.max(succ_vals) * 100 if len(succ_vals) > 0 else 0.0
        latest_succ = succ_vals[-1] * 100 if len(succ_vals) > 0 else 0.0
        return latest_step, latest_ret, best_ret, latest_succ, peak_succ

    fun_step, fun_lret, fun_bret, fun_lsucc, fun_bsucc = get_stats(data_fun)
    cat_step, cat_lret, cat_bret, cat_lsucc, cat_bsucc = get_stats(data_cat)

    print(f"{'指標 (Metric)':<26} | {'Standard FuN':<18} | {'FeUdalCat':<18}")
    print("-" * 70)
    print(f"{'當前訓練步數 (Steps)':<26} | {fun_step:<18,} | {cat_step:<18,}")
    print(f"{'最新評估回報 (Latest Return)':<24} | {fun_lret:<18.2f} | {cat_lret:<18.2f}")
    print(f"{'歷史最佳回報 (Peak Return)':<24} | {fun_bret:<18.2f} | {cat_bret:<18.2f}")
    print(f"{'最新成功率 (Latest Success)':<24} | {f'{fun_lsucc:.1f}%':<18} | {f'{cat_lsucc:.1f}%':<18}")
    print(f"{'最高成功率 (Peak Success)':<24} | {f'{fun_bsucc:.1f}%':<18} | {f'{cat_bsucc:.1f}%':<18}")
    print("=" * 70)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)
    colors = {"fun": "#1f77b4", "cat": "#d62728"}

    # Subplot 1: Deterministic Return
    ax1 = axes[0]
    ax1.set_title("Deterministic Evaluation Return\n(平均評估回報對比)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Environment Steps (環境步數)", fontsize=11)
    ax1.set_ylabel("Mean Episode Return", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    if "eval/mean_reward" in data_fun:
        s, v = data_fun["eval/mean_reward"]
        ax1.plot(s, v, color=colors["fun"], alpha=0.3, linewidth=1)
        ax1.plot(s, smooth_curve(v), color=colors["fun"], linewidth=2.2, label=f"Standard FuN (Max: {fun_bret:.1f})")

    if "eval/mean_reward" in data_cat:
        s, v = data_cat["eval/mean_reward"]
        ax1.plot(s, v, color=colors["cat"], alpha=0.3, linewidth=1)
        ax1.plot(s, smooth_curve(v), color=colors["cat"], linewidth=2.2, label=f"FeUdalCat (Max: {cat_bret:.1f})")

    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    # Subplot 2: Success Rate
    ax2 = axes[1]
    ax2.set_title("Maze Completion Success Rate (%)\n(迷宮抵達終點成功率對比)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Environment Steps (環境步數)", fontsize=11)
    ax2.set_ylabel("Success Rate (%)", fontsize=11)
    ax2.set_ylim(-2, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)

    if "eval/success_rate" in data_fun:
        s, v = data_fun["eval/success_rate"]
        ax2.plot(s, v * 100, color=colors["fun"], linewidth=2.2, marker="o", markersize=4, label=f"Standard FuN (Peak: {fun_bsucc:.1f}%)")

    if "eval/success_rate" in data_cat:
        s, v = data_cat["eval/success_rate"]
        ax2.plot(s, v * 100, color=colors["cat"], linewidth=2.2, marker="s", markersize=4, label=f"FeUdalCat (Peak: {cat_bsucc:.1f}%)")

    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"\n[+] 對比圖表已成功輸出至: {args.output}")


if __name__ == "__main__":
    main()
