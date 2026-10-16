import argparse
import os
import time
import imageio
import numpy as np
import torch
from torch.distributions import Categorical

from envs import make_atari_env, resolve_env_id
from feudalnet import FeUdalNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained FeUdal Network on Atari")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint .pt file")
    parser.add_argument("--env-name", type=str, default="ALE/MontezumaRevenge-v5",
                        help="Gymnasium Atari environment ID")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Max steps per evaluation episode")
    parser.add_argument("--record-gif", type=str, default="montezuma_play.gif",
                        help="Output GIF filepath (set to empty to disable)")
    parser.add_argument("--time-horizon", type=int, default=10,
                        help="Manager temporal horizon (c)")
    parser.add_argument("--dilation", type=int, default=10,
                        help="Manager Dilated LSTM dilation radius (r)")
    parser.add_argument("--hidden-dim-manager", type=int, default=256,
                        help="Manager latent space hidden dimension (d)")
    parser.add_argument("--hidden-dim-worker", type=int, default=16,
                        help="Worker goal projection dimension (k)")
    parser.add_argument("--human", action="store_true",
                        help="Display real-time interactive game window on screen")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy action selection instead of sampling")
    return parser.parse_args()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    actual_id = resolve_env_id(args.env_name)
    render_mode = "human" if args.human else ("rgb_array" if args.record_gif else None)
    env_thunk = make_atari_env(actual_id, seed=0, render_mode=render_mode)
    env = env_thunk()

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    model = FeUdalNetwork(
        num_workers=1,
        input_shape=obs_shape,
        n_actions=num_actions,
        hidden_dim_manager=args.hidden_dim_manager,
        hidden_dim_worker=args.hidden_dim_worker,
        time_horizon=args.time_horizon,
        dilation=args.dilation,
        eps=0.0,
        device=device
    ).to(device)
    model.eval()

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "rms_mean" in checkpoint:
            model.preprocessor.rms.mean = checkpoint["rms_mean"]
            model.preprocessor.rms.var = checkpoint["rms_var"]
            model.preprocessor.rms.count = checkpoint["rms_count"]
        print("Checkpoint successfully loaded!")
    else:
        print("No checkpoint specified; evaluating randomly initialized agent.")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        goals, states, masks = model.init_obj()
        # Shape batch dimension for single worker: [1, H, W, C]
        obs_batch = np.expand_dims(obs, axis=0)

        total_reward = 0.0
        frames = []
        step = 0

        # Initial frame capture
        frame = env.unwrapped.render()
        if frame is not None and args.record_gif:
            frames.append(frame)

        for step in range(args.max_steps):
            with torch.no_grad():
                action_probs, goals, states, value_m, value_w = model(
                    obs_batch, goals, states, masks[-1]
                )

                if args.deterministic:
                    action = torch.argmax(action_probs, dim=-1).item()
                else:
                    dist = Categorical(action_probs)
                    action = dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            obs_batch = np.expand_dims(obs, axis=0)
            total_reward += reward

            done = terminated or truncated
            mask = torch.tensor([[0.0 if done else 1.0]], dtype=torch.float32, device=device)
            masks.pop(0)
            masks.append(mask)

            if args.record_gif:
                frame = env.unwrapped.render()
                if frame is not None:
                    frames.append(frame)

            if args.human:
                time.sleep(1.0 / 30)

            if done:
                break

        print(f"Episode {ep + 1}/{args.episodes} finished | Steps: {step + 1} | Total Reward: {total_reward}")

        if args.record_gif and len(frames) > 0:
            gif_filename = f"eval_ep{ep + 1}_{args.record_gif}"
            print(f"Saving episode replay animation to: {gif_filename} ({len(frames)} frames)...")
            imageio.mimsave(gif_filename, frames[::2], fps=15)
            print("GIF saved successfully!")

    env.close()


if __name__ == "__main__":
    evaluate(parse_args())
