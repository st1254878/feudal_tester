#!/usr/bin/env python3
"""
=============================================================================
AntMaze 雙軌並行訓練啟動器 (FuN vs FeUdalCat Dual Launcher)
=============================================================================
功能：
  針對本機 14 核心 / 20 執行緒 CPU + RTX 4060 Ti GPU 進行雙任務算力分配：
  - 各分配 8 個並行環境 (num_workers=8, AsyncVectorEnv)，合計 16 個 MuJoCo 模擬行程
  - 加上 2 個主訓練行程，合計 18 個行程，完美適配 20 個邏輯執行緒，防止排程顛簸
  - 各維持 num_steps=250，完整涵蓋迷宮全程路徑進行信用分配
  - 獨立紀錄 TensorBoard 與 Checkpoint，互不干擾

使用方式：
  1. 獨立視窗模式 (預設，在 Windows 上開啟兩個獨立終端視窗，方便即時觀察兩者進度)：
     python launch_dual.py

  2. 背景背景執行模式 (輸出日誌至 logs/fun.log 與 logs/feudal_cat.log)：
     python launch_dual.py --mode background
=============================================================================
"""

import os
import sys
import glob
import argparse
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser(description="AntMaze FuN vs FeUdalCat Dual Parallel Training Launcher")
    parser.add_argument("--env-name", type=str, default="AntMaze_UMazeDense-v4",
                        help="Gymnasium environment ID (default: AntMaze_UMazeDense-v4)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Workers per model (default: 8; 8 * 2 = 16 workers total)")
    parser.add_argument("--num-steps", type=int, default=250,
                        help="Rollout unroll steps K (default: 250)")
    parser.add_argument("--max-steps", type=int, default=int(3e6),
                        help="Total environment steps per run (default: 3M)")
    parser.add_argument("--eval-interval", type=int, default=25000,
                        help="Evaluation step interval (default: 25k)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--success-bonus", type=float, default=50.0,
                        help="Goal arrival bonus (default: 50.0)")
    parser.add_argument("--mode", type=str, choices=["window", "background"], default="window" if sys.platform == "win32" else "background",
                        help="Launch mode: 'window' (open separate console windows) or 'background' (redirect to logs/)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--time-horizon", type=int, default=50,
                        help="Manager temporal horizon (c, default: 50)")
    parser.add_argument("--dilation", type=int, default=50,
                        help="Manager Dilated LSTM dilation radius (r, default: 50)")
    parser.add_argument("--reward-mode", type=str, default="sparse", choices=["sparse", "progress", "dense"],
                        help="Reward formulation (default: 'sparse')")
    parser.add_argument("--freeze-worker", action="store_true", default=False,
                        help="Freeze Worker policy and Percept module weights (default: False)")
    parser.add_argument("--pretrained-model", type=str, default=None,
                        help="Path to pre-trained checkpoint to load (default: None)")
    parser.add_argument("--pretrained-fun", type=str, default=None,
                        help="Path to pre-trained checkpoint specifically for FuN (default: None)")
    parser.add_argument("--pretrained-cat", type=str, default=None,
                        help="Path to pre-trained checkpoint specifically for FeUdalCat (default: None)")
    parser.add_argument("--pretrained-73m", action="store_true", default=False,
                        help="Auto-load the 73.7M step pre-trained checkpoints for FuN and FeUdalCat")
    parser.add_argument("--reset-manager", action="store_true", default=False,
                        help="Reset Manager module after loading pre-trained weights (default: False)")
    parser.add_argument("--eps", type=float, default=0.15,
                        help="Exploration probability for random Manager goals (default: 0.15)")
    parser.add_argument("--undiluted-reward", action=argparse.BooleanOptionalAction, default=True,
                        help="Preserve full gradient scale of rare reward events across parallel workers (default: True)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Automatically resume both models from their latest checkpoint in checkpoints_continuous/")
    return parser.parse_args()


def find_latest_checkpoint(pattern, root="checkpoints_continuous"):
    candidates = []
    for d in glob.glob(os.path.join(root, "*")):
        if os.path.isdir(d) and pattern in os.path.basename(d).lower():
            latest_pt = os.path.join(d, "model_latest.pt")
            if os.path.isfile(latest_pt):
                candidates.append((os.path.getmtime(latest_pt), latest_pt))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build_cmd(model_type, run_name, args, python_exe, resume_path=None, pretrained_path=None):
    cmd = [
        python_exe,
        "train_continuous.py",
        "--env-name", args.env_name,
        "--model-type", model_type,
        "--run-name", run_name,
        "--num-workers", str(args.num_workers),
        "--num-steps", str(args.num_steps),
        "--max-steps", str(args.max_steps),
        "--time-horizon", str(args.time_horizon),
        "--dilation", str(args.dilation),
        "--reward-mode", args.reward_mode,
        "--eps", str(args.eps),
        "--eval-interval", str(args.eval_interval),
        "--eval-episodes", str(args.eval_episodes),
        "--success-bonus", str(args.success_bonus),
        "--seed", str(args.seed),
        "--geodesic",
        "--terminate-on-success",
        "--async-envs"
    ]
    if not args.undiluted_reward:
        cmd.append("--no-undiluted-reward")
    if args.freeze_worker:
        cmd.append("--freeze-worker")
    target_pre = pretrained_path if pretrained_path else args.pretrained_model
    if target_pre:
        cmd.extend(["--pretrained-model", target_pre])
    if args.reset_manager:
        cmd.append("--reset-manager")
    if resume_path:
        cmd.extend(["--resume", resume_path])
    return cmd


def main():
    args = parse_args()
    workspace_dir = os.path.abspath(os.path.dirname(__file__))

    # Detect python executable in venv
    venv_python = os.path.join(workspace_dir, ".venv", "Scripts", "python.exe")
    if not os.path.isfile(venv_python):
        venv_python = sys.executable

    print("=" * 70)
    print("🚀 啟動 AntMaze 雙軌並行訓練 (FuN vs FeUdalCat)")
    print("=" * 70)
    print(f"環境: {args.env_name}")
    print(f"每模型並行 Worker 數: {args.num_workers} (兩模型合計 {args.num_workers * 2} 個 AsyncVectorEnv)")
    print(f"Rollout 步數 (num_steps): {args.num_steps} (完整涵蓋 0~終點路徑)")
    print(f"總訓練步數: {args.max_steps:,} 步 | 評估間隔: {args.eval_interval:,} 步")
    print(f"終點額外獎勵: +{args.success_bonus:.1f} | 測地線獎勵: 已啟用")
    print(f"執行模式: {args.mode}")
    print("=" * 70)

    fun_resume = None
    cat_resume = None
    if args.resume:
        fun_resume = find_latest_checkpoint("fun")
        cat_resume = find_latest_checkpoint("cat")
        if fun_resume:
            print(f"📦 [FuN] 接續訓練來源: {fun_resume}")
        else:
            print("⚠️ [FuN] 未找到先前 Checkpoint，將從頭開始訓練。")
        if cat_resume:
            print(f"📦 [FeUdalCat] 接續訓練來源: {cat_resume}")
        else:
            print("⚠️ [FeUdalCat] 未找到先前 Checkpoint，將從頭開始訓練。")

    default_73m_fun = "checkpoints_continuous/20261026_030411_cont_antmaze_fun_c50_bonus50_100m/model_latest.pt"
    default_73m_cat = "checkpoints_continuous/20261026_030433_cont_antmaze_cat_c50_bonus50_100m/model_latest.pt"

    pretrained_fun = args.pretrained_fun or (default_73m_fun if args.pretrained_73m else args.pretrained_model)
    pretrained_cat = args.pretrained_cat or (default_73m_cat if args.pretrained_73m else args.pretrained_model)

    if pretrained_fun:
        print(f"🎯 [FuN] 載入預訓練權重: {pretrained_fun} (凍結 Worker: {args.freeze_worker}, 重置 Manager: {args.reset_manager})")
    if pretrained_cat:
        print(f"🎯 [FeUdalCat] 載入預訓練權重: {pretrained_cat} (凍結 Worker: {args.freeze_worker}, 重置 Manager: {args.reset_manager})")

    fun_cmd = build_cmd("fun", f"antmaze_fun_{args.reward_mode}", args, venv_python, resume_path=fun_resume, pretrained_path=pretrained_fun)
    cat_cmd = build_cmd("feudal_cat", f"antmaze_feudal_cat_{args.reward_mode}", args, venv_python, resume_path=cat_resume, pretrained_path=pretrained_cat)

    if args.mode == "window" and sys.platform == "win32":
        # Launch each run in its own Command Prompt window with distinct titles
        fun_str = " ".join(f'"{c}"' if " " in c else c for c in fun_cmd)
        cat_str = " ".join(f'"{c}"' if " " in c else c for c in cat_cmd)

        fun_window_cmd = f'start "AntMaze - Standard FuN" {fun_str}'
        cat_window_cmd = f'start "AntMaze - FeUdalCat" {cat_str}'

        print("[1/2] 正在開啟 Standard FuN 訓練視窗...")
        subprocess.Popen(fun_window_cmd, shell=True, cwd=workspace_dir)
        time.sleep(2)  # Stagger initializations to avoid resource race

        print("[2/2] 正在開啟 FeUdalCat 訓練視窗...")
        subprocess.Popen(cat_window_cmd, shell=True, cwd=workspace_dir)

        print("\n✅ 兩大模型訓練視窗已成功開啟！")
        print("💡 您可以隨時執行下列命令監看兩者進度與繪製對比圖：")
        print("   - 終端進度速查: python check_progress_continuous.py")
        print("   - 繪製對比圖表: python compare_runs.py")
        print("=" * 70)

    else:
        # Background subprocess mode with log redirection
        os.makedirs(os.path.join(workspace_dir, "logs"), exist_ok=True)
        fun_log_path = os.path.join(workspace_dir, "logs", "train_fun.log")
        cat_log_path = os.path.join(workspace_dir, "logs", "train_feudal_cat.log")

        print(f"[1/2] 正在背景啟動 Standard FuN (日誌: logs/train_fun.log)...")
        fun_log = open(fun_log_path, "w", encoding="utf-8")
        p_fun = subprocess.Popen(fun_cmd, stdout=fun_log, stderr=subprocess.STDOUT, cwd=workspace_dir)
        time.sleep(2)

        print(f"[2/2] 正在背景啟動 FeUdalCat (日誌: logs/train_feudal_cat.log)...")
        cat_log = open(cat_log_path, "w", encoding="utf-8")
        p_cat = subprocess.Popen(cat_cmd, stdout=cat_log, stderr=subprocess.STDOUT, cwd=workspace_dir)

        print(f"\n✅ 兩大模型已在背景運行 (PID FuN: {p_fun.pid}, PID FeUdalCat: {p_cat.pid})")
        print("按 Ctrl+C 可安全中斷兩者訓練。\n")

        try:
            while p_fun.poll() is None or p_cat.poll() is None:
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n偵測到中斷訊號，正在安全終止雙軌進程...")
            p_fun.terminate()
            p_cat.terminate()
            p_fun.wait()
            p_cat.wait()
            print("雙軌進程已全數終止。")
        finally:
            fun_log.close()
            cat_log.close()


if __name__ == "__main__":
    main()
