#!/usr/bin/env python3
"""
=============================================================================
AntMaze / RL 訓練圖表繪製模板 (Single Run Plotter: Return & Success Rate Only)
=============================================================================
功能：
  僅繪製 Evaluation Return (平均回報) 與 Success Rate (%) 兩項核心指標。

使用方式：
  1. 直接在最上方 [CONFIG 快速設定區] 修改參數後執行：
         python plot_single_run.py

  2. 或透過終端命令列傳入參數：
         # 自動繪製最新的 run：
         python plot_single_run.py --latest

         # 指定特定 run 目錄：
         python plot_single_run.py --run_dir runs_continuous/20261026_030411_cont_antmaze_fun_c50_bonus50_100m

         # 若該次訓練是接續 (resume) 前一次 checkpoint，可傳入前期目錄自動拼接：
         python plot_single_run.py --run_dir runs_continuous/20261026_030411_cont_antmaze_fun_c50_bonus50_100m --resumed_from runs_continuous/20261025_004816_cont_antmaze_fun_c50_bonus50_20m
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 設定 Windows 終端輸出編碼相容性
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 設定字型支援與負號
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# ⚙️ [CONFIG 快速設定區]：不想打指令時，直接在此修改即可！
# =============================================================================
CONFIG = {
    # 訓練記錄目錄 (留空 "" 會自動抓取 runs_root 下最新修改的 run)
    "run_dir": "",

    # 前期訓練目錄 (若從某個 checkpoint 接續訓練，填入前期目錄自動無縫拼接；無則留 "")
    "resumed_from": "",

    # 搜尋 runs 的根目錄
    "runs_root": "runs_continuous",

    # 圖片輸出資料夾與檔名 (檔名留空 "" 會自動命名為 <run名稱>_eval.png)
    "output_dir": "plots",
    "output_filename": "",

    # 高斯平滑強度 sigma (數值越大越平滑，0 代表純原始數據)
    "smooth_sigma": 6.0,

    # 曲線主題顏色 (例如 FuN 紅色 '#d62728'，或經典藍 '#1f77b4'，綠色 '#2ca02c')
    "theme_color": "#d62728",

    # 圖片解析度 DPI
    "dpi": 200,
}
# =============================================================================


def gaussian_smooth(y, sigma=3.0):
    """一維高斯平滑濾波器，保持邊緣特徵"""
    if len(y) == 0 or sigma <= 0.0:
        return np.array(y)
    truncate = 4.0
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / (sigma ** 2) * (x ** 2))
    phi_x = phi_x / phi_x.sum()
    y_padded = np.pad(y, (radius, radius), mode='edge')
    smoothed = np.convolve(y_padded, phi_x, mode='valid')
    return smoothed


def find_latest_run(root_dir):
    """自動尋找 root_dir 下最新修改且包含 tfevents 的 run 目錄"""
    if not os.path.isdir(root_dir):
        return None
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d))]
    valid_runs = []
    for d in subdirs:
        if any(f.startswith('events.out.tfevents') for f in os.listdir(d)):
            valid_runs.append(d)
    if not valid_runs:
        return None
    valid_runs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return valid_runs[0]


def load_eval_scalars(run_dir):
    """僅讀取評估指標 (eval/mean_reward 與 eval/success_rate)，讀取極快"""
    if not run_dir or not os.path.isdir(run_dir):
        return {}
    ef_list = [f for f in os.listdir(run_dir) if f.startswith('events.out.tfevents')]
    if not ef_list:
        print(f"[!] 警告: 在 {run_dir} 找不到任何 events.out.tfevents 檔案")
        return {}
    ef_path = os.path.join(run_dir, ef_list[0])
    print(f"[+] 正在讀取評估數據: {os.path.basename(run_dir)}")
    ea = event_accumulator.EventAccumulator(ef_path)
    ea.Reload()

    scalar_data = {}
    available_tags = ea.Tags().get('scalars', [])
    target_tags = ['eval/mean_reward', 'eval/reward', 'eval/success_rate']
    for tag in target_tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            scalar_data[tag] = (np.array(steps), np.array(vals))
    return scalar_data


def merge_series(d1, d2, tag):
    """合併兩段接續訓練的特定 tag 數據並按 step 遞增去重"""
    s1, v1 = d1.get(tag, (np.array([]), np.array([])))
    s2, v2 = d2.get(tag, (np.array([]), np.array([])))

    if len(s1) == 0:
        return s2, v2
    if len(s2) == 0:
        return s1, v1

    combined = list(zip(s1, v1)) + list(zip(s2, v2))
    combined.sort(key=lambda x: x[0])

    u_steps, u_vals = [], []
    seen = set()
    for s, v in combined:
        if s not in seen:
            seen.add(s)
            u_steps.append(s)
            u_vals.append(v)
    return np.array(u_steps), np.array(u_vals)


def plot_eval_curves(run_dir, resumed_from="", output_dir="plots", output_filename="",
                     smooth_sigma=6.0, theme_color="#d62728", dpi=200):
    """主要繪圖邏輯：僅輸出 Return 與 Success Rate 兩格圖表"""
    run_name = os.path.basename(os.path.normpath(run_dir))
    print(f"\n=======================================================")
    print(f"[+] 開始繪製評估曲線: {run_name}")
    print(f"=======================================================")

    # 讀取 TB 資料
    d_resumed = load_eval_scalars(resumed_from) if resumed_from else {}
    d_current = load_eval_scalars(run_dir)

    if not d_current and not d_resumed:
        print("[x] 錯誤: 沒有讀取到任何評估數據，終止繪圖。")
        return None

    # 定位回報與成功率
    tag_ret = 'eval/mean_reward' if ('eval/mean_reward' in d_current or 'eval/mean_reward' in d_resumed) else 'eval/reward'
    tag_suc = 'eval/success_rate'

    s_ret, v_ret = merge_series(d_resumed, d_current, tag_ret)
    s_suc, v_suc = merge_series(d_resumed, d_current, tag_suc)

    if len(s_ret) == 0 and len(s_suc) == 0:
        print("[x] 錯誤: 未找到 eval/mean_reward 或 eval/success_rate 標籤。")
        return None

    # 建立 2 行 1 列圖表 (上下疊放，共享 X 軸)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=dpi, sharex=True)

    # 主標題
    title_text = f"Evaluation Progress: {run_name}"
    if resumed_from:
        title_text += f"\n(Resumed from: {os.path.basename(os.path.normpath(resumed_from))})"
    fig.suptitle(title_text, fontsize=15, fontweight='bold', y=0.98)

    # -------------------------------------------------------------
    # (a) Deterministic Evaluation Return
    # -------------------------------------------------------------
    if len(s_ret) > 0:
        ax1.plot(s_ret, v_ret, color=theme_color, alpha=0.18, linestyle='--', linewidth=0.8, label='Raw Return')
        v_ret_smooth = gaussian_smooth(v_ret, sigma=smooth_sigma)
        
        peak_idx = np.argmax(v_ret)
        peak_step = s_ret[peak_idx]
        peak_ret = v_ret[peak_idx]
        latest_ret = v_ret[-1]

        ax1.plot(s_ret, v_ret_smooth, color=theme_color, linewidth=2.8,
                 label=f'Smoothed Return (σ={smooth_sigma}) | Peak: {peak_ret:.2f} | Latest: {latest_ret:.2f}')

        ax1.scatter([peak_step], [peak_ret], color=theme_color, s=80, zorder=5)
        ax1.annotate(f'Peak Return: {peak_ret:.2f}\n(Step {peak_step:,})',
                     xy=(peak_step, peak_ret),
                     xytext=(peak_step - (s_ret[-1] * 0.18), peak_ret * 0.93),
                     arrowprops=dict(arrowstyle='->', color=theme_color, lw=1.6),
                     fontweight='bold', color=theme_color, fontsize=10)

        if resumed_from and len(d_resumed.get(tag_ret, ([], []))[0]) > 0:
            resume_step = d_resumed[tag_ret][0][-1]
            ax1.axvline(x=resume_step, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            ax1.text(resume_step * 1.02, peak_ret * 0.95, 'Checkpoint Resume ->',
                     color='#444444', fontsize=9.5, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

        ax1.set_ylabel('Deterministic Mean Return', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Evaluation Return', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    else:
        ax1.text(0.5, 0.5, 'No Evaluation Return Data Found', ha='center', va='center')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # -------------------------------------------------------------
    # (b) Deterministic Evaluation Success Rate (%)
    # -------------------------------------------------------------
    if len(s_suc) > 0:
        suc_pct = v_suc * 100.0
        ax2.plot(s_suc, suc_pct, color=theme_color, alpha=0.18, linestyle=':', linewidth=0.8, label='Raw Success Rate')
        v_suc_smooth = gaussian_smooth(suc_pct, sigma=smooth_sigma)

        peak_idx = np.argmax(suc_pct)
        peak_step = s_suc[peak_idx]
        peak_suc = suc_pct[peak_idx]
        latest_suc = suc_pct[-1]

        total_checks = len(v_suc)
        succ_checks = np.sum(v_suc > 0)
        succ_episodes = np.sum(np.round(v_suc * 20))
        check_rate = (succ_checks / total_checks * 100.0) if total_checks > 0 else 0.0

        label_suc = (f'Smoothed Success Rate (σ={smooth_sigma}) | '
                     f'Peak: {peak_suc:.1f}% | Total Reached: {succ_episodes:.0f} eps '
                     f'({succ_checks}/{total_checks} checks, {check_rate:.1f}%)')
        ax2.plot(s_suc, v_suc_smooth, color=theme_color, linewidth=2.8, label=label_suc)

        if peak_suc > 0:
            ax2.scatter([peak_step], [peak_suc], color=theme_color, s=80, zorder=5)
            ax2.annotate(f'Peak Success: {peak_suc:.1f}%\n(Step {peak_step:,})',
                         xy=(peak_step, peak_suc),
                         xytext=(peak_step - (s_suc[-1] * 0.18), peak_suc * 0.88),
                         arrowprops=dict(arrowstyle='->', color=theme_color, lw=1.6),
                         fontweight='bold', color=theme_color, fontsize=10)

        if resumed_from and len(d_resumed.get(tag_suc, ([], []))[0]) > 0:
            resume_step = d_resumed[tag_suc][0][-1]
            ax2.axvline(x=resume_step, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        ax2.set_xlabel('Environment Steps', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Evaluation Success Rate (%)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', frameon=True, framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, 'No Evaluation Success Rate Data Found', ha='center', va='center')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 儲存圖片
    os.makedirs(output_dir, exist_ok=True)
    if not output_filename:
        output_filename = f"{run_name}_eval.png"
    if not output_filename.endswith('.png'):
        output_filename += '.png'
    save_path = os.path.join(output_dir, output_filename)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    total_steps = s_ret[-1] if len(s_ret) > 0 else (s_suc[-1] if len(s_suc) > 0 else 0)
    print(f"[OK] 訓練步數: {total_steps:,} 步 (~{total_steps/1e6:.1f}M)")
    if len(s_ret) > 0:
        print(f"[OK] 最高回報 (Peak Return): {peak_ret:.2f} (最新: {latest_ret:.2f})")
    if len(s_suc) > 0:
        print(f"[OK] 最高成功率 (Peak Success): {peak_suc:.1f}% (累計通關: {succ_episodes:.0f} 回合)")
    print(f"[OK] 圖片已儲存至: {save_path}")
    print(f"=======================================================\n")
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="AntMaze 評估曲線繪製 (Return & Success Rate)")
    parser.add_argument("--run_dir", type=str, default=CONFIG["run_dir"],
                        help="訓練日誌目錄 (例如 runs_continuous/xxxx)")
    parser.add_argument("--resumed_from", type=str, default=CONFIG["resumed_from"],
                        help="前期接續訓練目錄 (若有的話)")
    parser.add_argument("--latest", action="store_true",
                        help="自動選取最新的訓練目錄")
    parser.add_argument("--runs_root", type=str, default=CONFIG["runs_root"],
                        help="尋找 runs 的根目錄")
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"],
                        help="輸出圖表資料夾")
    parser.add_argument("--output_filename", type=str, default=CONFIG["output_filename"],
                        help="輸出檔名 (預設為 <run_name>_eval.png)")
    parser.add_argument("--smooth", type=float, default=CONFIG["smooth_sigma"],
                        help="平滑強度 sigma (預設 6.0)")
    parser.add_argument("--color", type=str, default=CONFIG["theme_color"],
                        help="曲線主題顏色 (預設 #d62728)")
    parser.add_argument("--dpi", type=int, default=CONFIG["dpi"],
                        help="圖片解析度 (預設 200)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_run = args.run_dir

    if args.latest or not target_run:
        latest_dir = find_latest_run(args.runs_root)
        if latest_dir:
            print(f"[+] 未指定 --run_dir，自動選取最新 run: {latest_dir}")
            target_run = latest_dir
        else:
            print(f"[x] 在 {args.runs_root} 下找不到包含 TensorBoard 檔案的目錄！")
            sys.exit(1)

    plot_eval_curves(
        run_dir=target_run,
        resumed_from=args.resumed_from,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        smooth_sigma=args.smooth,
        theme_color=args.color,
        dpi=args.dpi,
    )
