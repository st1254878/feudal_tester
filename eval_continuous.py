import argparse
import os
import time
import imageio
import numpy as np
import torch
import gymnasium as gym

from feudalnet_continuous import ContinuousFeUdalNetwork
from feudal_cat_continuous import ContinuousFeUdalCatNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Continuous FeUdal Network on MuJoCo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint .pt file")
    parser.add_argument("--model-type", type=str, default="fun", choices=["fun", "feudal_cat"],
                        help="Model architecture: standard FuN ('fun') or direct goal concatenation ('feudal_cat')")
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v4",
                        help="Gymnasium MuJoCo environment ID (HalfCheetah-v4, Ant-v4, etc.)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per evaluation episode")
    parser.add_argument("--record-gif", type=str, default="cheetah_play.gif",
                        help="Output GIF filepath (set to empty to disable)")
    parser.add_argument("--human", action="store_true",
                        help="Display real-time interactive 3D physics window on screen")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic mean action instead of sampling")
    parser.add_argument("--time-horizon", type=int, default=10,
                        help="Manager temporal horizon (c)")
    parser.add_argument("--dilation", type=int, default=10,
                        help="Manager Dilated LSTM dilation radius (r)")
    parser.add_argument("--hidden-dim-manager", type=int, default=256,
                        help="Manager latent space hidden dimension (d)")
    parser.add_argument("--hidden-dim-worker", type=int, default=16,
                        help="Worker goal projection dimension (k)")
    return parser.parse_args()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running continuous evaluation on device: {device}")

    render_mode = "human" if args.human else ("rgb_array" if args.record_gif else None)
    env = gym.make(args.env_name, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model_cls = ContinuousFeUdalCatNetwork if args.model_type == "feudal_cat" else ContinuousFeUdalNetwork
    model = model_cls(
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
    model.eval()

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        print("Checkpoint successfully loaded!")
    else:
        print("No checkpoint specified; evaluating randomly initialized agent.")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        goals, states, masks = model.init_obj()
        obs_batch = np.expand_dims(obs, axis=0)

        total_reward = 0.0
        frames = []
        step = 0

        frame = env.render() if args.record_gif else None
        if frame is not None:
            frames.append(frame)

        for step in range(args.max_steps):
            with torch.no_grad():
                action_dist, goals, states, value_m, value_w = model(
                    obs_batch, goals, states, masks[-1]
                )

                if args.deterministic:
                    action = action_dist.mean.cpu().numpy()[0]
                else:
                    action = action_dist.sample().cpu().numpy()[0]

            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_batch = np.expand_dims(obs, axis=0)
            total_reward += reward

            done = terminated or truncated
            mask = torch.tensor([[0.0 if done else 1.0]], dtype=torch.float32, device=device)
            masks.pop(0)
            masks.append(mask)

            if args.record_gif:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if args.human:
                time.sleep(1.0 / 60)

            if done:
                break

        print(f"Episode {ep + 1}/{args.episodes} finished | Steps: {step + 1} | Total Reward: {total_reward:.2f}")

        if args.record_gif and len(frames) > 0:
            gif_filename = f"eval_ep{ep + 1}_{args.record_gif}"
            print(f"Saving episode replay animation to: {gif_filename} ({len(frames)} frames)...")
            imageio.mimsave(gif_filename, frames[::2], fps=30)
            print("Animation saved successfully!")

    env.close()


if __name__ == "__main__":
    evaluate(parse_args())
