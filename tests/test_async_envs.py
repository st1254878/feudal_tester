import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gymnasium as gym
import numpy as np
import torch

from train_continuous import make_mujoco_vec_envs


def test_sync_vs_async_creation():
    """Verify both SyncVectorEnv and AsyncVectorEnv instantiate properly."""
    print("Testing SyncVectorEnv instantiation...")
    sync_envs = make_mujoco_vec_envs(
        "AntMaze_UMazeDense-v4",
        num_envs=4,
        seed=101,
        async_envs=False
    )
    assert isinstance(sync_envs, gym.vector.SyncVectorEnv), f"Expected SyncVectorEnv, got {type(sync_envs)}"
    obs, info = sync_envs.reset(seed=101)
    assert obs.shape == (4, 31), f"Unexpected obs shape: {obs.shape}"
    actions = sync_envs.action_space.sample()
    next_obs, rewards, terms, truncs, infos = sync_envs.step(actions)
    assert next_obs.shape == (4, 31)
    assert rewards.shape == (4,)
    sync_envs.close()
    print("PASS: test_sync_vs_async_creation (SyncVectorEnv)")

    print("Testing AsyncVectorEnv instantiation...")
    async_envs = make_mujoco_vec_envs(
        "AntMaze_UMazeDense-v4",
        num_envs=4,
        seed=102,
        async_envs=True
    )
    assert isinstance(async_envs, gym.vector.AsyncVectorEnv), f"Expected AsyncVectorEnv, got {type(async_envs)}"
    obs, info = async_envs.reset(seed=102)
    assert obs.shape == (4, 31), f"Unexpected obs shape: {obs.shape}"
    actions = async_envs.action_space.sample()
    next_obs, rewards, terms, truncs, infos = async_envs.step(actions)
    assert next_obs.shape == (4, 31)
    assert rewards.shape == (4,)
    async_envs.close()
    print("PASS: test_sync_vs_async_creation (AsyncVectorEnv)")


def test_async_16_workers_stepping():
    """Verify 16 parallel workers execute steps, rewards, and resets reliably."""
    print("Testing 16 parallel workers with AsyncVectorEnv...")
    envs = make_mujoco_vec_envs(
        "AntMaze_UMazeDense-v4",
        num_envs=16,
        seed=42,
        async_envs=True
    )
    obs, _ = envs.reset(seed=42)
    assert obs.shape == (16, 31)

    for step in range(25):
        actions = np.zeros((16, 8), dtype=np.float32)
        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        assert next_obs.shape == (16, 31)
        assert rewards.shape == (16,)
        assert terms.shape == (16,)
        assert truncs.shape == (16,)

    envs.close()
    print("PASS: test_async_16_workers_stepping")


def test_full_rollout_with_storage():
    """Verify 250-step rollout with Storage and ContinuousFeUdalNetwork runs without shape mismatch."""
    from feudalnet_continuous import ContinuousFeUdalNetwork
    from storage import Storage

    num_workers = 4
    num_steps = 250
    obs_dim = 31
    act_dim = 8
    device = torch.device("cpu")

    model = ContinuousFeUdalNetwork(
        num_workers=num_workers,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device
    )
    storage = Storage(size=num_steps)
    goals, states, masks = model.init_obj()

    dummy_obs = np.zeros((num_workers, obs_dim), dtype=np.float32)

    for _ in range(num_steps):
        action_dist, goals, states, value_m, value_w = model(
            dummy_obs, goals, states, masks[-1]
        )
        actions = action_dist.sample()
        rewards = np.zeros(num_workers, dtype=np.float32)
        masks.pop(0)
        mask = torch.ones(num_workers, 1, dtype=torch.float32, device=device)
        masks.append(mask)

        storage.add({
            "r": torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1),
            "r_i": model.intrinsic_reward(states, goals, masks),
            "v_w": value_w,
            "v_m": value_m,
            "logp": torch.zeros(num_workers, 1, device=device),
            "entropy": torch.zeros(num_workers, 1, device=device),
            "s_goal_cos": model.state_goal_cosine(states, goals, masks),
            "m": mask
        })

    assert len(storage.r) == num_steps, f"Expected {num_steps} rollout steps, got {len(storage.r)}"
    print(f"PASS: test_full_rollout_with_storage (Successfully stored {num_steps} steps)")


if __name__ == "__main__":
    test_sync_vs_async_creation()
    test_async_16_workers_stepping()
    test_full_rollout_with_storage()
    print("\nALL ASYNC & NUM_STEPS TESTS PASSED! (3/3)")
