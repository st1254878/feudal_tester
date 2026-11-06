import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from feudalnet_continuous import ContinuousFeUdalNetwork
from feudal_cat_continuous import ContinuousFeUdalCatNetwork


def run_evaluation_suite(
    checkpoint_path: str,
    env_name: str,
    model_type: str,
    num_episodes: int = 20,
    base_seed: int = 1000,
    device: str = "cuda"
):
    print(f"\n{'='*70}")
    print(f"Evaluating: {env_name} | Architecture: {model_type.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dummy_env = gym.make(env_name)
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]
    dummy_env.close()

    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    if isinstance(ckpt_args, dict):
        hidden_dim_manager = ckpt_args.get("hidden_dim_manager", 256)
        hidden_dim_worker = ckpt_args.get("hidden_dim_worker", 16)
        time_horizon = ckpt_args.get("time_horizon", 10)
        dilation = ckpt_args.get("dilation", 10)
    else:
        hidden_dim_manager = getattr(ckpt_args, "hidden_dim_manager", 256)
        hidden_dim_worker = getattr(ckpt_args, "hidden_dim_worker", 16)
        time_horizon = getattr(ckpt_args, "time_horizon", 10)
        dilation = getattr(ckpt_args, "dilation", 10)

    model_cls = ContinuousFeUdalCatNetwork if model_type == "feudal_cat" else ContinuousFeUdalNetwork

    model = model_cls(
        num_workers=1,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim_manager=hidden_dim_manager,
        hidden_dim_worker=hidden_dim_worker,
        time_horizon=time_horizon,
        dilation=dilation,
        eps=0.0,
        device=dev
    ).to(dev)

    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    test_configs = [
        {"name": "Normal (Baseline)", "ablate_goal": "none", "deterministic": True},
        {"name": "Zero Goal (Det)", "ablate_goal": "zero", "deterministic": True},
        {"name": "Zero Goal (Sampled)", "ablate_goal": "zero", "deterministic": False},
        {"name": "Random Goal (Det)", "ablate_goal": "random", "deterministic": True},
    ]

    results = {}

    for cfg in test_configs:
        cfg_name = cfg["name"]
        ablate_mode = cfg["ablate_goal"]
        det = cfg["deterministic"]

        ep_returns = []
        ep_lengths = []
        ep_distances = []
        act_norms = []

        env = gym.make(env_name)

        with torch.no_grad():
            for ep in range(num_episodes):
                obs, _ = env.reset(seed=base_seed + ep)
                x_start = float(env.unwrapped.data.qpos[0])

                goals, states, masks = model.init_obj()
                worker_hidden_dim = model.worker.wrnn.hidden_size
                model.worker_hidden = (
                    torch.zeros(1, worker_hidden_dim, device=dev),
                    torch.zeros(1, worker_hidden_dim, device=dev)
                )
                model.manager_hidden = model.manager.mrnn.init_hidden(1, dev)

                obs_batch = np.expand_dims(obs, axis=0)
                ep_return = 0.0
                ep_len = 0
                ep_act_norms = []

                done = False
                while not done:
                    action_dist, goals, states, _, _ = model(
                        obs_batch, goals, states, masks[-1], save=True, ablate_goal=ablate_mode
                    )

                    if det:
                        action = action_dist.mean.cpu().numpy()[0]
                    else:
                        action = action_dist.sample().cpu().numpy()[0]

                    ep_act_norms.append(float(np.linalg.norm(action)))
                    action = np.clip(action, -1.0, 1.0)

                    obs, reward, terminated, truncated, _ = env.step(action)
                    obs_batch = np.expand_dims(obs, axis=0)
                    ep_return += reward
                    ep_len += 1

                    done = terminated or truncated
                    mask = torch.tensor([[0.0 if done else 1.0]], dtype=torch.float32, device=dev)
                    masks.pop(0)
                    masks.append(mask)

                x_end = float(env.unwrapped.data.qpos[0])
                distance = x_end - x_start

                ep_returns.append(ep_return)
                ep_lengths.append(ep_len)
                ep_distances.append(distance)
                act_norms.append(np.mean(ep_act_norms))

        env.close()

        results[cfg_name] = {
            "mean_return": float(np.mean(ep_returns)),
            "std_return": float(np.std(ep_returns)),
            "min_return": float(np.min(ep_returns)),
            "max_return": float(np.max(ep_returns)),
            "mean_length": float(np.mean(ep_lengths)),
            "std_length": float(np.std(ep_lengths)),
            "mean_distance": float(np.mean(ep_distances)),
            "std_distance": float(np.std(ep_distances)),
            "mean_act_norm": float(np.mean(act_norms)),
            "all_returns": ep_returns
        }

        print(f"  [{cfg_name:18s}] Return: {results[cfg_name]['mean_return']:8.2f} ± {results[cfg_name]['std_return']:6.2f} | "
              f"Distance: {results[cfg_name]['mean_distance']:7.2f}m | "
              f"Len: {results[cfg_name]['mean_length']:6.1f} | "
              f"ActNorm: {results[cfg_name]['mean_act_norm']:5.2f}")

    return results


def main():
    experiments = [
        {
            "key": "cheetah_fun",
            "env": "HalfCheetah-v4",
            "model_type": "fun",
            "checkpoint": "checkpoints_continuous/20261018_021526_cont_cheetah_fun_10m/model_best.pt"
        },
        {
            "key": "cheetah_cat",
            "env": "HalfCheetah-v4",
            "model_type": "feudal_cat",
            "checkpoint": "checkpoints_continuous/20261018_021543_cont_cheetah_cat_10m/model_best.pt"
        },
        {
            "key": "ant_fun",
            "env": "Ant-v4",
            "model_type": "fun",
            "checkpoint": "checkpoints_continuous/20261018_114728_cont_ant_fun_20m/model_best.pt"
        },
        {
            "key": "ant_cat",
            "env": "Ant-v4",
            "model_type": "feudal_cat",
            "checkpoint": "checkpoints_continuous/20261018_114744_cont_ant_cat_20m/model_best.pt"
        }
    ]

    all_results = {}
    for exp in experiments:
        res = run_evaluation_suite(
            checkpoint_path=exp["checkpoint"],
            env_name=exp["env"],
            model_type=exp["model_type"],
            num_episodes=20,
            base_seed=1000
        )
        all_results[exp["key"]] = res

    os.makedirs("plots", exist_ok=True)
    with open("plots/manager_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print("ALL ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Saved results to plots/manager_ablation_results.json")
    print("="*70)


if __name__ == "__main__":
    main()
