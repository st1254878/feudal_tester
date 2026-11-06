import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import gymnasium as gym
try:
    import gymnasium_robotics
except ImportError:
    pass

from maze_wrappers import AntMazeGeodesicWrapper
from feudalnet_continuous import ContinuousFeUdalNetwork
from feudal_cat_continuous import ContinuousFeUdalCatNetwork


def test_geodesic_monotonicity():
    """Verify that corridor navigation monotonically decreases geodesic distance and increases reward."""
    raw_env = gym.make("AntMaze_UMazeDense-v4")
    env = AntMazeGeodesicWrapper(raw_env, success_bonus=50.0, max_corridor_dist=25.0)

    goal_xy = np.array([-4.0, -4.0])
    corridor_waypoints = [
        np.array([-4.0, 4.0]),  # Top-left (start)
        np.array([0.0, 4.0]),   # Top-mid (heading to turn)
        np.array([4.0, 4.0]),   # Top-right (turn corner)
        np.array([4.0, 0.0]),   # Mid-right (downward hall)
        np.array([4.0, -4.0]),  # Bottom-right
        np.array([0.0, -4.0]),  # Bottom-mid
        np.array([-4.0, -4.0])  # Goal
    ]

    dists = [env.compute_geodesic_distance(p, goal_xy) for p in corridor_waypoints]
    print("\nGeodesic distances along corridor:", dists)

    # Verify strict monotonic decrease
    for i in range(len(dists) - 1):
        assert dists[i] > dists[i + 1], (
            f"Distance failed to decrease: step {i} ({dists[i]}) vs step {i+1} ({dists[i+1]})"
        )
    assert dists[-1] == 0.0, f"Distance at goal should be 0, got {dists[-1]}"
    env.close()


def test_geodesic_wall_trap_avoidance():
    """Verify that hitting the wall between halls gives higher distance (lower reward) than heading to the turn."""
    raw_env = gym.make("AntMaze_UMazeDense-v4")
    env = AntMazeGeodesicWrapper(raw_env, success_bonus=50.0, max_corridor_dist=25.0)

    goal_xy = np.array([-4.0, -4.0])
    wall_pt = np.array([-4.0, 0.0])       # Dead-end wall dividing the two hallways
    turn_pt = np.array([0.0, 4.0])        # Correct direction towards open hallway

    dist_wall = env.compute_geodesic_distance(wall_pt, goal_xy)
    dist_turn = env.compute_geodesic_distance(turn_pt, goal_xy)

    print(f"\nDist to goal from wall trap: {dist_wall:.2f}m vs from turn: {dist_turn:.2f}m")
    assert dist_wall > dist_turn, (
        f"Wall trap distance ({dist_wall:.2f}) should be greater than turn ({dist_turn:.2f})"
    )
    env.close()


def test_success_bonus_and_termination():
    """Verify success bonus and termination upon goal touch."""
    raw_env = gym.make("AntMaze_UMazeDense-v4")
    env = AntMazeGeodesicWrapper(raw_env, success_bonus=50.0, terminate_on_success=True)

    obs, info = env.reset(seed=42)
    assert not env.has_succeeded
    assert not info["success"]

    # Manually place ant at desired goal position
    goal_pos = env.unwrapped.goal
    qpos = np.concatenate([goal_pos, [0.75], [1.0, 0.0, 0.0, 0.0], np.zeros(8)])
    qvel = np.zeros(14)
    env.unwrapped.ant_env.set_state(qpos, qvel)

    action = np.zeros(8, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert info["success"], "Expected success to be True"
    assert info["success_bonus_awarded"], "Expected success_bonus_awarded to be True"
    assert reward >= 50.0, f"Expected reward >= 50.0, got {reward}"
    assert terminated, "Expected terminated to be True when terminate_on_success=True"

    # Step again: bonus should not be awarded twice
    obs2, reward2, term2, trunc2, info2 = env.step(action)
    assert not info2["success_bonus_awarded"], "Bonus should only be awarded once per episode"
    assert reward2 < 50.0, f"Reward on subsequent step should not include bonus: {reward2}"
    env.close()


def test_spatial_obs_protection():
    """Verify that observation normalization preserves spatial coordinates in [-1, 1] without running mean drift."""
    for model_cls in [ContinuousFeUdalNetwork, ContinuousFeUdalCatNetwork]:
        model = model_cls(num_workers=4, obs_dim=31, act_dim=8, device=torch.device("cpu"))
        model.train()

        # Batch 1: Simulated Ant in top-left
        obs1 = torch.zeros(4, 31)
        obs1[:, 0:2] = torch.tensor([-4.0, 4.0])   # Ant position
        obs1[:, 2:4] = torch.tensor([-4.0, -4.0])  # Desired goal
        obs1[:, 4:] = torch.randn(4, 27)           # Proprioception

        norm_obs1 = model._normalize_obs(obs1, update=True)
        # Verify coordinates are exactly pos * 0.2
        assert torch.allclose(norm_obs1[:, 0:2], torch.tensor([[-0.8, 0.8]])), f"Spatial pos corrupted for {model_cls.__name__}"
        assert torch.allclose(norm_obs1[:, 2:4], torch.tensor([[-0.8, -0.8]])), f"Spatial goal corrupted for {model_cls.__name__}"

        # Batch 2: Different ant positions - ensure spatial coords are not shifted by running mean
        obs2 = torch.zeros(4, 31)
        obs2[:, 0:2] = torch.tensor([4.0, -4.0])   # Ant position moved
        obs2[:, 2:4] = torch.tensor([-4.0, -4.0])  # Desired goal
        obs2[:, 4:] = torch.randn(4, 27)

        norm_obs2 = model._normalize_obs(obs2, update=True)
        assert torch.allclose(norm_obs2[:, 0:2], torch.tensor([[0.8, -0.8]])), f"Running mean shifted spatial coords for {model_cls.__name__}"
        assert torch.allclose(norm_obs2[:, 2:4], torch.tensor([[-0.8, -0.8]])), f"Running mean shifted goal coords for {model_cls.__name__}"


if __name__ == "__main__":
    test_geodesic_monotonicity()
    print("PASS: test_geodesic_monotonicity")
    test_geodesic_wall_trap_avoidance()
    print("PASS: test_geodesic_wall_trap_avoidance")
    test_success_bonus_and_termination()
    print("PASS: test_success_bonus_and_termination")
    test_spatial_obs_protection()
    print("PASS: test_spatial_obs_protection")
    print("\nALL GEODESIC REWARD & BONUS TESTS PASSED! (4/4)")
