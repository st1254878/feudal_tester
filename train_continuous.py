import argparse
import os
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics

from feudalnet_continuous import ContinuousFeUdalNetwork, continuous_feudal_loss
from feudal_cat_continuous import ContinuousFeUdalCatNetwork, continuous_feudal_cat_loss
from storage import Storage
from logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Continuous FeUdal Networks (FuN) Training on MuJoCo")

    # Environment & System
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v4",
                        help="Gymnasium MuJoCo environment ID (HalfCheetah-v4, Ant-v4, etc.)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel environments to run (SyncVectorEnv)")
    parser.add_argument("--num-steps", type=int, default=200,
                        help="Rollout steps (n-step unroll length K)")
    parser.add_argument("--max-steps", type=int, default=int(3e6),
                        help="Maximum total environment steps to train (default: 3M steps)")
    parser.add_argument("--cuda", type=bool, default=True,
                        help="Enable CUDA acceleration if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--run-name", type=str, default="fun_halfcheetah",
                        help="Run experiment name for logging")

    # Optimization
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="Base learning rate for Worker & Percept modules")
    parser.add_argument("--lr-manager", type=float, default=0.0001,
                        help="Learning rate for Manager module (two-timescale stability, default: 1e-4)")
    parser.add_argument("--lr-decay", action=argparse.BooleanOptionalAction, default=True,
                        help="Linearly decay learning rates over max-steps (default: True, use --no-lr-decay to disable)")
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="Gradient clipping maximum norm")
    parser.add_argument("--entropy-coef", type=float, default=0.001,
                        help="Entropy coefficient for continuous exploration")

    # FuN Hyperparameters
    parser.add_argument("--time-horizon", type=int, default=10,
                        help="Manager temporal horizon (c)")
    parser.add_argument("--dilation", type=int, default=10,
                        help="Manager Dilated LSTM dilation radius (r)")
    parser.add_argument("--hidden-dim-manager", type=int, default=256,
                        help="Manager latent space hidden dimension (d)")
    parser.add_argument("--hidden-dim-worker", type=int, default=16,
                        help="Worker goal projection dimension (k)")
    parser.add_argument("--gamma-w", type=float, default=0.99,
                        help="Discount factor for Worker")
    parser.add_argument("--gamma-m", type=float, default=0.995,
                        help="Discount factor for Manager")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Worker intrinsic reward coefficient alpha in [0, 1]")
    parser.add_argument("--eps", type=float, default=1e-4,
                        help="Exploration probability for random Manager goals")

    # Checkpoint & Logging
    parser.add_argument("--save-interval", type=int, default=100000,
                        help="Step interval to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training")

    # Periodic Evaluation
    parser.add_argument("--eval-interval", type=int, default=30000,
                        help="Step interval to run deterministic policy evaluation (default: 30k, 0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes per evaluation check (default: 10)")

    # Model architecture selection
    parser.add_argument("--model-type", type=str, default="fun", choices=["fun", "feudal_cat"],
                        help="Model architecture: standard FuN ('fun') or direct goal concatenation ('feudal_cat')")

    return parser.parse_args()


def make_mujoco_vec_envs(env_id: str, num_envs: int, seed: int = 0):
    def make_single(idx):
        def _thunk():
            env = gym.make(env_id)
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed + idx)
            return env
        return _thunk

    return gym.vector.SyncVectorEnv([make_single(i) for i in range(num_envs)])


def sample_action(dist):
    action = dist.sample()
    # Continuous MuJoCo actions are clamped to [-1, 1]
    clamped_action = torch.clamp(action, -1.0, 1.0)
    logp = dist.log_prob(action).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    return clamped_action.cpu().numpy(), logp, entropy


def evaluate_policy(eval_model, train_model, env_id: str, num_episodes: int = 5, seed: int = 1000, device = "cuda"):
    """
    Evaluates policy with deterministic mean actions (no exploration noise)
    across a fixed set of episodes using a dedicated 1-worker eval model.
    """
    eval_model.load_state_dict(train_model.state_dict())
    eval_model.eval()

    eval_env = gym.make(env_id)
    returns = []
    lengths = []

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, _ = eval_env.reset(seed=seed + ep)
            goals, states, masks = eval_model.init_obj()
            # Reset recurrent memory for clean episode evaluation
            eval_model.worker_hidden = (
                torch.zeros(1, eval_model.k * eval_model.act_dim, device=device),
                torch.zeros(1, eval_model.k * eval_model.act_dim, device=device)
            )
            eval_model.manager_hidden = eval_model.manager.mrnn.init_hidden(1, device)

            obs_batch = np.expand_dims(obs, axis=0)
            ep_return = 0.0
            ep_len = 0

            done = False
            while not done:
                action_dist, goals, states, _, _ = eval_model(
                    obs_batch, goals, states, masks[-1]
                )
                action = action_dist.mean.cpu().numpy()[0]
                action = np.clip(action, -1.0, 1.0)

                obs, reward, terminated, truncated, _ = eval_env.step(action)
                obs_batch = np.expand_dims(obs, axis=0)
                ep_return += reward
                ep_len += 1

                done = terminated or truncated
                mask = torch.tensor([[0.0 if done else 1.0]], dtype=torch.float32, device=device)
                masks.pop(0)
                masks.append(mask)

            returns.append(ep_return)
            lengths.append(ep_len)

    eval_env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    mean_len = float(np.mean(lengths))
    return mean_ret, std_ret, mean_len





def train(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    # Use separate continuous logs/checkpoints to prevent collision with Atari runs
    os.makedirs("runs_continuous", exist_ok=True)
    os.makedirs("checkpoints_continuous", exist_ok=True)

    logger = Logger(
        run_name=f"cont_{args.run_name}",
        args=args,
        log_dir_base="runs_continuous",
        checkpoint_dir_base="checkpoints_continuous"
    )

    print(f"==================================================")
    print(f"Starting Continuous FeUdal Networks training")
    print(f"Environment: {args.env_name}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if use_cuda else 'CPU'})")
    print(f"Parallel Workers: {args.num_workers}")
    print(f"Rollout Steps (K): {args.num_steps}")
    print(f"Manager Horizon (c): {args.time_horizon}, Dilation (r): {args.dilation}")
    print(f"TensorBoard: {logger.log_dir}")
    print(f"Checkpoints: {logger.checkpoint_dir}")
    print(f"==================================================")

    envs = make_mujoco_vec_envs(args.env_name, args.num_workers, seed=args.seed)

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    if args.model_type == "feudal_cat":
        model_cls = ContinuousFeUdalCatNetwork
        loss_fn = continuous_feudal_cat_loss
        print("[Model] Using FeUdalCat (Direct Goal Concatenation)")
    else:
        model_cls = ContinuousFeUdalNetwork
        loss_fn = continuous_feudal_loss
        print("[Model] Using Standard FuN (Direction Modulation)")

    model = model_cls(
        num_workers=args.num_workers,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim_manager=args.hidden_dim_manager,
        hidden_dim_worker=args.hidden_dim_worker,
        time_horizon=args.time_horizon,
        dilation=args.dilation,
        eps=args.eps,
        device=device
    ).to(device)

    eval_model = model_cls(
        num_workers=1,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim_manager=args.hidden_dim_manager,
        hidden_dim_worker=args.hidden_dim_worker,
        time_horizon=args.time_horizon,
        dilation=args.dilation,
        eps=0.0,
        device=device
    ).to(device)

    # Two-timescale Adam optimizer:
    # Worker & Percept use args.lr (default 3e-4); Manager uses args.lr_manager (default 1e-4)
    # to prevent Manager latent goal drift ("Moving Target" instability).
    worker_params = list(model.percept.parameters()) + list(model.worker.parameters())
    manager_params = list(model.manager.parameters())
    optimizer = torch.optim.Adam([
        {"params": worker_params, "lr": args.lr, "initial_lr": args.lr, "name": "worker"},
        {"params": manager_params, "lr": args.lr_manager, "initial_lr": args.lr_manager, "name": "manager"},
    ], eps=1e-5)

    step = 0
    best_eval_reward = -float("inf")
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError:
            print("[Warning] Optimizer param groups changed; re-initializing optimizer state.")
        step = checkpoint.get("step", 0)
        best_eval_reward = checkpoint.get("best_eval_reward", -float("inf"))
        print(f"Resumed from step {step}")

    goals, states, masks = model.init_obj()
    obs, info = envs.reset(seed=args.seed)

    storage = Storage(size=args.num_steps)
    next_save_step = step + args.save_interval
    next_eval_step = step + args.eval_interval if args.eval_interval > 0 else float("inf")

    try:
        while step < args.max_steps:
            # Linear Learning Rate Decay down to 10% of initial LR
            if args.lr_decay:
                progress = min(1.0, step / max(1, args.max_steps))
                lr_mult = max(0.1, 1.0 - 0.9 * progress)
                optimizer.param_groups[0]["lr"] = args.lr * lr_mult
                optimizer.param_groups[1]["lr"] = args.lr_manager * lr_mult

            model.repackage_hidden()
            goals = [g.detach() for g in goals]
            storage.reset()

            for _ in range(args.num_steps):
                action_dist, goals, states, value_m, value_w = model(
                    obs, goals, states, masks[-1]
                )

                actions, logp, entropy = sample_action(action_dist)
                next_obs, reward, terminated, truncated, info = envs.step(actions)
                done = np.logical_or(terminated, truncated)

                logger.log_episode(info, step)

                mask = torch.tensor(1.0 - done.astype(float), dtype=torch.float32, device=device).unsqueeze(-1)
                masks.pop(0)
                masks.append(mask)

                storage.add({
                    "r": torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(-1),
                    "r_i": model.intrinsic_reward(states, goals, masks),
                    "v_w": value_w,
                    "v_m": value_m,
                    "logp": logp.unsqueeze(-1),
                    "entropy": entropy.unsqueeze(-1),
                    "s_goal_cos": model.state_goal_cosine(states, goals, masks),
                    "m": mask
                })

                obs = next_obs
                step += args.num_workers

            with torch.no_grad():
                *_, next_v_m, next_v_w = model(obs, goals, states, masks[-1], save=False)
                next_v_m = next_v_m.detach()
                next_v_w = next_v_w.detach()

            optimizer.zero_grad()
            loss, loss_dict = loss_fn(storage, next_v_m, next_v_w, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_dict["train/lr_worker"] = optimizer.param_groups[0]["lr"]
            loss_dict["train/lr_manager"] = optimizer.param_groups[1]["lr"]
            logger.log_scalars(loss_dict, step)

            # Periodic Deterministic Evaluation
            if args.eval_interval > 0 and step >= next_eval_step:
                eval_mean, eval_std, eval_len = evaluate_policy(
                    eval_model, model, args.env_name, num_episodes=args.eval_episodes,
                    seed=args.seed + 1000 + (step // max(1, args.eval_interval)), device=device
                )
                logger.log_scalars({
                    "eval/mean_reward": eval_mean,
                    "eval/std_reward": eval_std,
                    "eval/mean_length": eval_len
                }, step)
                print(f"\n>>> [EVAL @ Step {step:,}] Deterministic Return: {eval_mean:.2f} +/- {eval_std:.2f} | Length: {eval_len:.1f}")
                if eval_mean > best_eval_reward:
                    best_eval_reward = eval_mean
                    best_path = os.path.join(logger.checkpoint_dir, "model_best.pt")
                    torch.save({
                        "step": step,
                        "best_eval_reward": best_eval_reward,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args)
                    }, best_path)
                    print(f">>> [NEW BEST] Saved peak model ({best_eval_reward:.2f}) -> {best_path}\n")
                else:
                    print()
                next_eval_step = step + args.eval_interval

            if step >= next_save_step:
                ckpt_dict = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args)
                }
                ckpt_path = os.path.join(logger.checkpoint_dir, f"model_step_{step}.pt")
                latest_path = os.path.join(logger.checkpoint_dir, "model_latest.pt")
                torch.save(ckpt_dict, ckpt_path)
                torch.save(ckpt_dict, latest_path)
                print(f"Saved checkpoint: {ckpt_path}")
                next_save_step = step + args.save_interval

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")

    finally:
        ckpt_dict = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args)
        }
        final_ckpt = os.path.join(logger.checkpoint_dir, f"model_final_{step}.pt")
        latest_path = os.path.join(logger.checkpoint_dir, "model_latest.pt")
        torch.save(ckpt_dict, final_ckpt)
        torch.save(ckpt_dict, latest_path)
        print(f"Final checkpoint saved: {final_ckpt}")
        envs.close()
        logger.close()


if __name__ == "__main__":
    train(parse_args())
