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
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of parallel environments to run (default: 16)")
    parser.add_argument("--num-steps", type=int, default=250,
                        help="Rollout steps (n-step unroll length K, default: 250)")
    parser.add_argument("--async-envs", action=argparse.BooleanOptionalAction, default=True,
                        help="Use AsyncVectorEnv for multi-process CPU parallelism (default: True, use --no-async-envs for SyncVectorEnv)")
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
    parser.add_argument("--time-horizon", type=int, default=50,
                        help="Manager temporal horizon (c, default: 50)")
    parser.add_argument("--dilation", type=int, default=50,
                        help="Manager Dilated LSTM dilation radius (r, default: 50)")
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
    parser.add_argument("--eps", type=float, default=0.15,
                        help="Exploration probability for random Manager goals (default: 0.15)")
    parser.add_argument("--undiluted-reward", action=argparse.BooleanOptionalAction, default=True,
                        help="Preserve full gradient scale of rare reward events across parallel workers (default: True)")

    # Reward Formulation & Navigation
    parser.add_argument("--reward-mode", type=str, default="sparse", choices=["sparse", "progress", "dense"],
                        help="Reward formulation: 'sparse' (0 until goal touch +50), 'progress' (geodesic distance delta + 50), or 'dense' (-d/25 penalty + 50)")
    parser.add_argument("--success-bonus", type=float, default=50.0,
                        help="Sparse completion bonus awarded once per episode upon reaching the goal (default: 50.0)")
    parser.add_argument("--geodesic", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable BFS geodesic corridor reward wrapper for AntMaze (default: True, use --no-geodesic to disable)")
    parser.add_argument("--terminate-on-success", action=argparse.BooleanOptionalAction, default=True,
                        help="Terminate episode immediately upon reaching the goal (default: True, use --no-terminate-on-success to disable)")

    # Hierarchical Pre-training & Freezing
    parser.add_argument("--freeze-worker", action=argparse.BooleanOptionalAction, default=False,
                        help="Freeze Worker policy and Percept module weights, training only Manager (default: False)")
    parser.add_argument("--pretrained-model", type=str, default=None,
                        help="Path to pre-trained checkpoint .pt file to initialize model weights")
    parser.add_argument("--reset-manager", action=argparse.BooleanOptionalAction, default=False,
                        help="Reset Manager module to initial random weights after loading pre-trained weights")

    # Checkpoint & Logging
    parser.add_argument("--save-interval", type=int, default=100000,
                        help="Step interval to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training")

    # Periodic Evaluation
    parser.add_argument("--eval-interval", type=int, default=30000,
                        help="Step interval to run deterministic policy evaluation (default: 30k, 0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of evaluation episodes per evaluation check (default: 20)")

    parser.add_argument("--spatial-obs-norm", type=str, default="fixed", choices=["fixed", "running"],
                        help="AntMaze position/goal input normalization: 'fixed' (x * 0.2) or 'running' (running mean/var, matches checkpoints trained before the fixed scaling)")

    # Model architecture selection
    parser.add_argument("--model-type", type=str, default="fun", choices=["fun", "feudal_cat", "worker_only"],
                        help="Model architecture: standard FuN ('fun'), direct goal concatenation ('feudal_cat'), or Worker-Only ('worker_only')")

    return parser.parse_args()


def make_mujoco_vec_envs(
    env_id: str,
    num_envs: int,
    seed: int = 0,
    reward_mode: str = "sparse",
    use_geodesic: bool = True,
    success_bonus: float = 50.0,
    terminate_on_success: bool = True,
    async_envs: bool = True
):
    def make_single(idx):
        def _thunk():
            try:
                import gymnasium_robotics
            except ImportError:
                pass
            env = gym.make(env_id)
            if "antmaze" in env_id.lower() and use_geodesic:
                from maze_wrappers import AntMazeGeodesicWrapper
                env = AntMazeGeodesicWrapper(
                    env,
                    reward_mode=reward_mode,
                    success_bonus=success_bonus,
                    terminate_on_success=terminate_on_success
                )
            if isinstance(env.observation_space, gym.spaces.Dict):
                from gymnasium.wrappers import FlattenObservation
                env = FlattenObservation(env)
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed + idx)
            return env
        return _thunk

    vec_cls = gym.vector.AsyncVectorEnv if (async_envs and num_envs > 1) else gym.vector.SyncVectorEnv
    # SAME_STEP: the obs returned on a done step is already the reset obs (what the training loop assumes);
    # the episode's last obs/info move to info["final_obs"] / info["final_info"].
    # The gymnasium default (NEXT_STEP) would insert one no-op transition per episode that leaks across episodes.
    return vec_cls([make_single(i) for i in range(num_envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def merge_final_info(info: dict, key: str):
    """
    Under SAME_STEP autoreset, envs that just finished report their reset info at the top level
    and their last-step info under info["final_info"]. Returns `key` with finished envs taken from final_info.
    """
    value = info.get(key)
    final_info = info.get("final_info", {})
    if key not in final_info:
        return value
    final_mask = final_info.get(f"_{key}", np.ones(len(final_info[key]), dtype=bool))
    if value is None:
        return np.where(final_mask, final_info[key], False)
    return np.where(final_mask, final_info[key], value)


def sample_action(dist):
    action = dist.sample()
    # Continuous MuJoCo actions are clamped to [-1, 1]
    clamped_action = torch.clamp(action, -1.0, 1.0)
    # Evaluate log_prob on the executed action to prevent gradient explosion on saturated bounds
    safe_action = torch.clamp(action, -0.999999, 0.999999)
    logp = dist.log_prob(safe_action).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    return clamped_action.cpu().numpy(), logp, entropy


def evaluate_policy(
    eval_model,
    train_model,
    env_id: str,
    num_episodes: int = 5,
    seed: int = 1000,
    device = "cuda",
    reward_mode: str = "sparse",
    use_geodesic: bool = True,
    success_bonus: float = 50.0,
    terminate_on_success: bool = True
):
    """
    Evaluates policy with deterministic mean actions (no exploration noise)
    across a fixed set of episodes using a dedicated 1-worker eval model.
    """
    try:
        import gymnasium_robotics
    except ImportError:
        pass
    eval_model.load_state_dict(train_model.state_dict(), strict=False)
    eval_model.eval()

    eval_env = gym.make(env_id)
    if "antmaze" in env_id.lower() and use_geodesic:
        from maze_wrappers import AntMazeGeodesicWrapper
        eval_env = AntMazeGeodesicWrapper(
            eval_env,
            reward_mode=reward_mode,
            success_bonus=success_bonus,
            terminate_on_success=terminate_on_success
        )
    if isinstance(eval_env.observation_space, gym.spaces.Dict):
        from gymnasium.wrappers import FlattenObservation
        eval_env = FlattenObservation(eval_env)

    returns = []
    lengths = []
    successes = []

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, _ = eval_env.reset(seed=seed + ep)
            goals, states, masks = eval_model.init_obj()
            # Reset recurrent memory for clean episode evaluation
            worker_hidden_dim = eval_model.worker.wrnn.hidden_size
            eval_model.worker_hidden = (
                torch.zeros(1, worker_hidden_dim, device=device),
                torch.zeros(1, worker_hidden_dim, device=device)
            )
            eval_model.manager_hidden = eval_model.manager.mrnn.init_hidden(1, device)

            obs_batch = np.expand_dims(obs, axis=0)
            ep_return = 0.0
            ep_len = 0
            ep_success = False

            done = False
            while not done:
                action_dist, goals, states, _, _ = eval_model(
                    obs_batch, goals, states, masks[-1]
                )
                action = action_dist.mean.cpu().numpy()[0]
                action = np.clip(action, -1.0, 1.0)

                obs, reward, terminated, truncated, info = eval_env.step(action)
                obs_batch = np.expand_dims(obs, axis=0)
                ep_return += reward
                ep_len += 1
                if info.get("success", False):
                    ep_success = True

                done = terminated or truncated
                mask = torch.tensor([[0.0 if done else 1.0]], dtype=torch.float32, device=device)
                masks.pop(0)
                masks.append(mask)

            returns.append(ep_return)
            lengths.append(ep_len)
            successes.append(1.0 if ep_success else 0.0)

    eval_env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    mean_len = float(np.mean(lengths))
    mean_success = float(np.mean(successes))
    return mean_ret, std_ret, mean_len, mean_success





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
    env_mode = "AsyncVectorEnv (Multiprocess)" if (args.async_envs and args.num_workers > 1) else "SyncVectorEnv (Single-thread)"
    print(f"Parallel Workers: {args.num_workers} [{env_mode}]")
    print(f"Rollout Steps (K): {args.num_steps}")
    print(f"Manager Horizon (c): {args.time_horizon}, Dilation (r): {args.dilation}")
    print(f"Reward Mode: {args.reward_mode.upper()} (Sparse Success Bonus: {args.success_bonus})")
    if args.freeze_worker:
        print("Worker Mode: FROZEN ?? (Training Manager only)")
    if args.pretrained_model:
        print(f"Pretrained Model: {args.pretrained_model} (Reset Manager: {args.reset_manager})")
    print(f"TensorBoard: {logger.log_dir}")
    print(f"Checkpoints: {logger.checkpoint_dir}")
    print(f"==================================================")

    envs = make_mujoco_vec_envs(
        args.env_name,
        args.num_workers,
        seed=args.seed,
        reward_mode=args.reward_mode,
        use_geodesic=args.geodesic,
        success_bonus=args.success_bonus,
        terminate_on_success=args.terminate_on_success,
        async_envs=args.async_envs
    )

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    if args.model_type == "feudal_cat":
        model_cls = ContinuousFeUdalCatNetwork
        loss_fn = continuous_feudal_cat_loss
        print("[Model] Using FeUdalCat (Direct Goal Concatenation)")
    elif args.model_type == "worker_only":
        from worker_only_continuous import ContinuousWorkerOnlyNetwork, continuous_worker_only_loss
        model_cls = ContinuousWorkerOnlyNetwork
        loss_fn = continuous_worker_only_loss
        print("[Model] Using Worker-Only (Flat Recurrent Actor-Critic)")
    else:
        model_cls = ContinuousFeUdalNetwork
        loss_fn = continuous_feudal_loss
        print("[Model] Using Standard FuN (Direction Modulation)")

    model_kwargs = {
        "num_workers": args.num_workers,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dim_manager": args.hidden_dim_manager,
        "hidden_dim_worker": args.hidden_dim_worker,
        "time_horizon": args.time_horizon,
        "dilation": args.dilation,
        "eps": args.eps,
        "device": device
    }
    if args.model_type != "worker_only":
        model_kwargs["spatial_obs_norm"] = args.spatial_obs_norm

    model = model_cls(**model_kwargs).to(device)

    eval_kwargs = dict(model_kwargs)
    eval_kwargs["num_workers"] = 1
    eval_kwargs["eps"] = 0.0
    eval_model = model_cls(**eval_kwargs).to(device)

    # Pretrained weights loading
    if args.pretrained_model and os.path.isfile(args.pretrained_model):
        print(f"[Pretrained] Loading weights from: {args.pretrained_model}")
        ckpt_data = torch.load(args.pretrained_model, map_location=device, weights_only=False)

        # Dynamic Worker dimension adaptation if checkpoint worker size differs (e.g. historical FeUdalCat runs)
        if "worker.wrnn.weight_hh" in ckpt_data["model"]:
            ckpt_hidden = ckpt_data["model"]["worker.wrnn.weight_hh"].shape[1]
            if hasattr(model, "worker_hidden_dim") and model.worker_hidden_dim != ckpt_hidden:
                from feudal_cat_continuous import ContinuousWorkerCat
                print(f"[Pretrained] Adapting Worker hidden dimension to checkpoint ({ckpt_hidden})...")
                model.worker_hidden_dim = ckpt_hidden
                model.worker = ContinuousWorkerCat(model.c, model.d, ckpt_hidden, model.act_dim, device).to(device)
                eval_model.worker_hidden_dim = ckpt_hidden
                eval_model.worker = ContinuousWorkerCat(eval_model.c, eval_model.d, ckpt_hidden, eval_model.act_dim, device).to(device)
                model.worker_hidden = (
                    torch.zeros(args.num_workers, ckpt_hidden, device=device),
                    torch.zeros(args.num_workers, ckpt_hidden, device=device)
                )
                eval_model.worker_hidden = (
                    torch.zeros(1, ckpt_hidden, device=device),
                    torch.zeros(1, ckpt_hidden, device=device)
                )

        model.load_state_dict(ckpt_data["model"], strict=False)
        eval_model.load_state_dict(ckpt_data["model"], strict=False)
        print("[Pretrained] Pretrained weights loaded successfully.")

        if args.reset_manager and hasattr(model, "manager"):
            print("[Pretrained] Resetting Manager to fresh initial random weights...")
            from feudalnet_continuous import ContinuousManager
            model.manager = ContinuousManager(args.time_horizon, args.hidden_dim_manager, args.dilation, args.eps, device).to(device)
            eval_model.manager = ContinuousManager(args.time_horizon, args.hidden_dim_manager, args.dilation, 0.0, device).to(device)
            model.manager_hidden = model.manager.mrnn.init_hidden(args.num_workers, device)
            eval_model.manager_hidden = eval_model.manager.mrnn.init_hidden(1, device)

    # Worker freezing
    if args.freeze_worker:
        print("[Freeze] Freezing Worker policy and Percept module (requires_grad = False)!")
        if hasattr(model, "worker"):
            model.worker.requires_grad_(False)
            eval_model.worker.requires_grad_(False)
        if hasattr(model, "percept"):
            model.percept.requires_grad_(False)
            eval_model.percept.requires_grad_(False)

    # Optimizer setup:
    if args.freeze_worker:
        optimizer = torch.optim.Adam([
            {"params": model.manager.parameters(), "lr": args.lr_manager, "initial_lr": args.lr_manager, "name": "manager"}
        ], eps=1e-5)
    elif args.model_type == "worker_only":
        optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": args.lr, "initial_lr": args.lr, "name": "worker"}
        ], eps=1e-5)
    else:
        worker_params = list(model.percept.parameters()) + list(model.worker.parameters())
        manager_params = list(model.manager.parameters())
        param_groups = [
            {"params": worker_params, "lr": args.lr, "initial_lr": args.lr, "name": "worker"},
            {"params": manager_params, "lr": args.lr_manager, "initial_lr": args.lr_manager, "name": "manager"},
        ]
        optimizer = torch.optim.Adam(param_groups, eps=1e-5)

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
        best_eval = checkpoint.get("best_eval_reward")
        best_eval_reward = best_eval if best_eval is not None else -float("inf")
        print(f"Resumed from step {step} (Best Eval Return: {best_eval_reward:.2f})")

    goals, states, masks = model.init_obj()
    obs, info = envs.reset(seed=args.seed)
    has_succeeded = np.zeros(args.num_workers, dtype=bool)

    storage = Storage(size=args.num_steps)
    # Critic targets are centered only when undiluted_reward is off; denormalize must match
    value_center = not args.undiluted_reward
    next_save_step = step + args.save_interval
    next_eval_step = step + args.eval_interval if args.eval_interval > 0 else float("inf")

    try:
        while step < args.max_steps:
            # Linear Learning Rate Decay down to 10% of initial LR
            if args.lr_decay:
                progress = min(1.0, step / max(1, args.max_steps))
                lr_mult = max(0.1, 1.0 - 0.9 * progress)
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * lr_mult

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

                # Time-limit truncation is not a true terminal: bootstrap gamma * V(final_obs).
                # Must run before reset_workers so the recurrent state still belongs to the finished episode.
                boot_m = torch.zeros(args.num_workers, 1, device=device)
                boot_w = torch.zeros(args.num_workers, 1, device=device)
                trunc_only = np.logical_and(truncated, np.logical_not(terminated))
                if trunc_only.any() and "final_obs" in info:
                    final_obs = np.stack([
                        np.asarray(info["final_obs"][w], dtype=np.float32) if trunc_only[w] else next_obs[w]
                        for w in range(args.num_workers)
                    ])
                    with torch.no_grad():
                        *_, v_m_final, v_w_final = model(final_obs, goals, states, torch.ones_like(masks[-1]), save=False)
                    trunc_t = torch.tensor(trunc_only, dtype=torch.float32, device=device).unsqueeze(-1)
                    boot_m = args.gamma_m * model.val_norm_m.denormalize(v_m_final, center=value_center) * trunc_t
                    boot_w = args.gamma_w * model.val_norm_w.denormalize(v_w_final, center=value_center) * trunc_t

                # Reset worker history immediately when an episode ends to prevent cross-episode contamination
                if done.any():
                    reset_mask = torch.tensor(1.0 - done.astype(float), dtype=torch.float32, device=device).unsqueeze(-1)
                    if hasattr(model, "reset_workers"):
                        model.reset_workers(reset_mask, goals, states)

                # Sparse Completion Bonus announcement & fallback
                succ = merge_final_info(info, "success")
                if succ is not None:
                    bonus_awarded = merge_final_info(info, "success_bonus_awarded")
                    for w in range(args.num_workers):
                        if succ[w] and not has_succeeded[w]:
                            if bonus_awarded is not None:
                                if not bonus_awarded[w] and args.success_bonus > 0:
                                    reward[w] += args.success_bonus
                            elif args.success_bonus > 0:
                                reward[w] += args.success_bonus
                            has_succeeded[w] = True
                            print(f">>> [GOAL REACHED] Worker {w} hit the goal at step {step:,}! Awarded +{args.success_bonus:.1f} bonus.")

                for w in range(args.num_workers):
                    if done[w]:
                        has_succeeded[w] = False

                # RecordEpisodeStatistics' "episode" entry lands in final_info under SAME_STEP autoreset
                logger.log_episode(info.get("final_info", {}), step)

                mask = torch.tensor(1.0 - done.astype(float), dtype=torch.float32, device=device).unsqueeze(-1)
                masks.pop(0)
                masks.append(mask)

                step_data = {
                    "r": torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(-1),
                    "r_i": model.intrinsic_reward(states, goals, masks),
                    "v_w": value_w,
                    "v_m": value_m,
                    "logp": logp.unsqueeze(-1),
                    "entropy": entropy.unsqueeze(-1),
                    "s_goal_cos": model.state_goal_cosine(states, goals, masks),
                    "m": mask,
                    "boot_m": boot_m,
                    "boot_w": boot_w
                }
                storage.add(step_data)

                obs = next_obs
                step += args.num_workers

            with torch.no_grad():
                *_, next_v_m_raw, next_v_w_raw = model(obs, goals, states, masks[-1], save=False)
                next_v_m = model.val_norm_m.denormalize(next_v_m_raw, center=value_center).detach()
                next_v_w = model.val_norm_w.denormalize(next_v_w_raw, center=value_center).detach()

            optimizer.zero_grad()
            loss, loss_dict = loss_fn(storage, next_v_m, next_v_w, model, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_dict["train/lr_worker"] = optimizer.param_groups[0]["lr"]
            if len(optimizer.param_groups) > 1:
                loss_dict["train/lr_manager"] = optimizer.param_groups[1]["lr"]
            logger.log_scalars(loss_dict, step)

            # Periodic Deterministic Evaluation
            if args.eval_interval > 0 and step >= next_eval_step:
                eval_mean, eval_std, eval_len, eval_success = evaluate_policy(
                    eval_model, model, args.env_name, num_episodes=args.eval_episodes,
                    seed=args.seed + 1000, device=device,
                    reward_mode=args.reward_mode,
                    use_geodesic=args.geodesic, success_bonus=args.success_bonus,
                    terminate_on_success=args.terminate_on_success
                )
                eval_metrics = {
                    "eval/mean_reward": eval_mean,
                    "eval/std_reward": eval_std,
                    "eval/mean_length": eval_len
                }
                if "maze" in args.env_name.lower():
                    eval_metrics["eval/success_rate"] = eval_success
                    print(f"\n>>> [EVAL @ Step {step:,}] Deterministic Return: {eval_mean:.2f} +/- {eval_std:.2f} | Length: {eval_len:.1f} | Success: {eval_success * 100:.1f}%")
                else:
                    print(f"\n>>> [EVAL @ Step {step:,}] Deterministic Return: {eval_mean:.2f} +/- {eval_std:.2f} | Length: {eval_len:.1f}")

                logger.log_scalars(eval_metrics, step)
                if eval_mean > best_eval_reward:
                    best_eval_reward = eval_mean
                    os.makedirs(logger.checkpoint_dir, exist_ok=True)
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
                    "best_eval_reward": best_eval_reward,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args)
                }
                os.makedirs(logger.checkpoint_dir, exist_ok=True)
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
            "best_eval_reward": best_eval_reward,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args)
        }
        os.makedirs(logger.checkpoint_dir, exist_ok=True)
        final_ckpt = os.path.join(logger.checkpoint_dir, f"model_final_{step}.pt")
        latest_path = os.path.join(logger.checkpoint_dir, "model_latest.pt")
        torch.save(ckpt_dict, final_ckpt)
        torch.save(ckpt_dict, latest_path)
        print(f"Final checkpoint saved: {final_ckpt}")
        envs.close()
        logger.close()


if __name__ == "__main__":
    import multiprocessing
    import sys
    multiprocessing.set_executable(sys.executable)
    train(parse_args())


