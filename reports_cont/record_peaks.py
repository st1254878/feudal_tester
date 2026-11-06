"""
Record a GIF of each run's peak checkpoint in every continuous environment.

For each checkpoint it first replays the training-time eval (20 deterministic episodes on seeds
seed+1000 .. seed+1019, same as evaluate_policy in train_continuous.py) to confirm the checkpoint
reproduces its logged score, then renders the best of those 20 episodes.

Writes reports/gifs/<env>_<model>.gif and prints a summary table.
Usage: python reports/record_peaks.py [--only name_substring]
"""
import argparse
import os
import sys

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from feudalnet_continuous import ContinuousFeUdalNetwork  # noqa: E402
from feudal_cat_continuous import ContinuousFeUdalCatNetwork  # noqa: E402
from worker_only_continuous import ContinuousWorkerOnlyNetwork  # noqa: E402

CK = os.path.join(ROOT, "checkpoints_continuous")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gifs")

# (gif name, env, checkpoint, model type, extra model kwargs, logged peak description)
PEAKS = [
    ("halfcheetah_fun", "HalfCheetah-v4", "20261018_021526_cont_cheetah_fun_10m/model_best.pt", "fun", {},
     "eval return 4,880"),
    ("halfcheetah_feudal_cat", "HalfCheetah-v4", "20261018_021543_cont_cheetah_cat_10m/model_best.pt", "feudal_cat", {},
     "eval return 3,401"),
    ("halfcheetah_worker_only", "HalfCheetah-v4", "20261019_231417_cont_cheetah_worker_only_10m/model_best.pt",
     "worker_only", {}, "eval return 5,057"),
    ("ant_fun", "Ant-v4", "20261018_114728_cont_ant_fun_20m/model_best.pt", "fun", {}, "eval return 4,843"),
    ("ant_feudal_cat", "Ant-v4", "20261018_114744_cont_ant_cat_20m/model_best.pt", "feudal_cat", {},
     "eval return 4,391"),
    ("ant_worker_only", "Ant-v4", "20261019_231437_cont_ant_worker_only_20m/model_best.pt", "worker_only", {},
     "eval return 4,173"),
    ("antmaze_fun", "AntMaze_UMazeDense-v4", "20261031_001629_cont_antmaze_fun_c50_bonus50_100m/model_step_81716800.pt",
     "fun", {"spatial_obs_norm": "running"}, "eval success 35%"),
    ("antmaze_feudal_cat", "AntMaze_UMazeDense-v4", "20261026_030433_cont_antmaze_cat_c50_bonus50_100m/model_best.pt",
     "feudal_cat", {"spatial_obs_norm": "running"}, "eval success 0% (never reached the goal)"),
]
MODEL_CLS = {"fun": ContinuousFeUdalNetwork, "feudal_cat": ContinuousFeUdalCatNetwork,
             "worker_only": ContinuousWorkerOnlyNetwork}
MODEL_NAME = {"fun": "FuN", "feudal_cat": "FeUdalCat", "worker_only": "Worker-only"}
FRAME_SKIP = 2          # keep every 2nd frame
FRAME_MS = 50           # 20 fps -> about 2x real time (sim dt = 0.05 s)
AFTER_SUCCESS = 40      # AntMaze: keep recording this many steps after the first goal touch


def make_env(env_id, render):
    try:
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
    except ImportError:
        pass
    kwargs = {"render_mode": "rgb_array", "width": 480, "height": 360} if render else {}
    env = gym.make(env_id, **kwargs)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    return env


def setup_camera(env, env_id):
    """
    Free camera (gymnasium resets the camera type to free on every render, so tracking is done by hand).
    Locomotion: side view that follows the torso. AntMaze: fixed overview from above.
    Returns a function to call before each render.
    """
    env.render()  # creates the offscreen viewer
    sim_env = getattr(env.unwrapped, "ant_env", env.unwrapped)  # AntMaze wraps an inner Ant env
    cam = sim_env.mujoco_renderer._get_viewer("rgb_array").cam
    if "maze" in env_id.lower():
        cam.lookat[:] = [0.0, 0.0, 0.0]
        cam.distance = 25.0
        cam.elevation = -65.0
        cam.azimuth = 90.0
        return lambda: None
    cam.distance = 5.0 if "cheetah" in env_id.lower() else 7.0
    cam.elevation = -20.0
    cam.azimuth = 90.0

    def follow():
        torso = sim_env.data.xpos[1]  # body 0 is the world
        cam.lookat[:] = [torso[0], torso[1], 0.5]
    follow()
    return follow


def load_model(path, model_type, extra, obs_dim, act_dim, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    model = MODEL_CLS[model_type](
        num_workers=1, obs_dim=obs_dim, act_dim=act_dim,
        hidden_dim_manager=a["hidden_dim_manager"], hidden_dim_worker=a["hidden_dim_worker"],
        time_horizon=a["time_horizon"], dilation=a["dilation"], eps=0.0, device=device, **extra
    ).to(device)
    # Older FeUdalCat checkpoints used a 16-unit Worker; the constructor now raises anything <= 16 to 128
    ck_hidden = ck["model"]["worker.wrnn.weight_hh"].shape[1]
    if model_type == "feudal_cat" and ck_hidden != model.worker_hidden_dim:
        from feudal_cat_continuous import ContinuousWorkerCat
        model.worker_hidden_dim = ck_hidden
        model.worker = ContinuousWorkerCat(model.c, model.d, ck_hidden, act_dim, device).to(device)
    result = model.load_state_dict(ck["model"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"  WARNING state_dict mismatch: missing={result.missing_keys} unexpected={result.unexpected_keys}")
    model.eval()
    return model, ck["step"], a.get("seed", 42)


def run_episode(model, env, seed, device, render=False, header="", env_id=""):
    obs, _ = env.reset(seed=seed)
    follow = setup_camera(env, env_id) if render else None
    goals, states, masks = model.init_obj()
    hd = model.worker.wrnn.hidden_size
    model.worker_hidden = (torch.zeros(1, hd, device=device), torch.zeros(1, hd, device=device))
    model.manager_hidden = model.manager.mrnn.init_hidden(1, device)
    total, success_step, frames, t = 0.0, None, [], 0
    done = False
    while not done:
        with torch.no_grad():
            dist, goals, states, _, _ = model(np.expand_dims(obs, 0), goals, states, masks[-1])
        action = np.clip(dist.mean.cpu().numpy()[0], -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        t += 1
        done = terminated or truncated
        masks.pop(0)
        masks.append(torch.tensor([[0.0 if done else 1.0]], device=device))
        if info.get("success", False) and success_step is None:
            success_step = t
        if render and t % FRAME_SKIP == 0:
            follow()
            frames.append(annotate(env.render(), header, t, total, success_step))
        if success_step is not None and t >= success_step + AFTER_SUCCESS and render:
            break
    return total, t, success_step, frames


def annotate(frame, header, t, total, success_step):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("segoeui.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    lines = [header, f"step {t}   return {total:,.1f}" + ("   GOAL REACHED" if success_step else "")]
    pad, y = 6, 6
    width = max(draw.textlength(s, font=font) for s in lines) + 2 * pad
    draw.rectangle([0, 0, width, 6 + 20 * len(lines)], fill=(0, 0, 0))
    for s in lines:
        draw.text((pad, y), s, fill=(255, 255, 255), font=font)
        y += 20
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cuda' matches the device the training-time evals ran on")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    summary = []
    for gif_name, env_id, rel, model_type, extra, logged in PEAKS:
        if args.only and args.only not in gif_name:
            continue
        path = os.path.join(CK, rel)
        print(f"\n== {gif_name}: {rel}")
        env = make_env(env_id, render=False)
        model, ck_step, seed = load_model(path, model_type, extra, env.observation_space.shape[0],
                                          env.action_space.shape[0], device)
        seeds = [seed + 1000 + i for i in range(20)]
        results = [run_episode(model, env, s, device) for s in seeds]
        env.close()
        rets = np.array([r[0] for r in results])
        succ = np.array([r[2] is not None for r in results])
        is_maze = "maze" in env_id.lower()
        if is_maze:
            # Best = successful episode that reached the goal fastest; else highest return
            ok = [i for i in range(20) if succ[i]]
            best = min(ok, key=lambda i: results[i][2]) if ok else int(rets.argmax())
            replay = f"success {succ.mean()*100:.0f}% (mean return {rets.mean():.1f})"
        else:
            best = int(rets.argmax())
            replay = f"mean return {rets.mean():,.0f} (min {rets.min():,.0f}, max {rets.max():,.0f})"
        print(f"  replayed 20 eval episodes: {replay}; logged peak: {logged}")

        header = f"{MODEL_NAME[model_type]} · {env_id} · checkpoint @ {ck_step/1e6:.2f}M steps · eval seed {seeds[best]}"
        env = make_env(env_id, render=True)
        total, length, success_step, frames = run_episode(model, env, seeds[best], device, render=True, header=header,
                                                          env_id=env_id)
        env.close()
        out = os.path.join(OUT, f"{gif_name}.gif")
        imageio.mimsave(out, frames, duration=FRAME_MS, loop=0)
        size_mb = os.path.getsize(out) / 1e6
        rerun_note = "" if abs(total - rets[best]) < 1e-3 or success_step else f" (re-run return {total:.1f})"
        print(f"  recorded seed {seeds[best]}: return {rets[best]:,.1f}, {length} steps"
              f"{', goal at step ' + str(success_step) if success_step else ''}{rerun_note} -> {out} ({size_mb:.1f} MB)")
        summary.append((gif_name, ck_step, logged, replay, seeds[best], rets[best], success_step, size_mb))

    print("\nSUMMARY")
    for row in summary:
        print(" | ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
