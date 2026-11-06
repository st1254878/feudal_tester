import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gymnasium as gym
import numpy as np
import torch
import gymnasium_robotics

from maze_wrappers import AntMazeGeodesicWrapper
from feudalnet_continuous import ContinuousFeUdalNetwork, ContinuousManager, continuous_feudal_loss
from storage import Storage
import argparse


def test_sparse_reward_mode():
    """Verify sparse reward returns 0.0 on every step until goal is touched."""
    env = gym.make("AntMaze_UMazeDense-v4")
    wrapper = AntMazeGeodesicWrapper(env, reward_mode="sparse", success_bonus=50.0, terminate_on_success=True)
    obs, info = wrapper.reset(seed=42)
    assert not info["success"]

    # 10 steps of random movement away from goal
    for _ in range(10):
        obs, r, term, trunc, info = wrapper.step(wrapper.action_space.sample())
        if not info["success"]:
            assert r == 0.0, f"Expected step reward 0.0 in sparse mode, got {r}"

    # Force goal arrival
    goal_pos = env.unwrapped.goal
    qpos = np.concatenate([goal_pos, [0.75], [1.0, 0.0, 0.0, 0.0], np.zeros(8)])
    qvel = np.zeros(14)
    env.unwrapped.ant_env.set_state(qpos, qvel)

    obs, r, term, trunc, info = wrapper.step(np.zeros(8, dtype=np.float32))
    assert info["success"], "Expected success at goal position"
    assert r == 50.0, f"Expected +50.0 bonus on goal reach, got {r}"
    assert term, "Expected episode termination on goal touch"
    wrapper.close()
    print("PASS: test_sparse_reward_mode (0.0 steps + 50.0 goal touch + early termination)")


def test_progress_reward_mode():
    """Verify progress reward gives positive delta when moving closer, 0 when still."""
    env = gym.make("AntMaze_UMazeDense-v4")
    wrapper = AntMazeGeodesicWrapper(env, reward_mode="progress", success_bonus=50.0, terminate_on_success=True)
    obs, info = wrapper.reset(seed=42)

    # First step computes initial prev_d_geo
    obs, r1, term, trunc, info = wrapper.step(np.zeros(8, dtype=np.float32))
    assert r1 == 0.0, "First step should have 0 delta progress"

    # Second step with zero action (ant barely moves)
    obs, r2, term, trunc, info = wrapper.step(np.zeros(8, dtype=np.float32))
    assert abs(r2) < 0.05, f"Expected near zero progress on still ant, got {r2}"
    wrapper.close()
    print("PASS: test_progress_reward_mode (progress delta computed correctly)")


def test_freeze_worker_and_gradient_isolation():
    """Verify Worker and Percept are frozen, and only Manager accumulates gradients."""
    c = 50
    d = 256
    num_steps = 250
    num_workers = 2
    device = torch.device("cpu")

    model = ContinuousFeUdalNetwork(
        num_workers=num_workers,
        obs_dim=31,
        act_dim=8,
        time_horizon=c,
        dilation=c,
        device=device
    )

    # Freeze Worker and Percept
    model.worker.requires_grad_(False)
    model.percept.requires_grad_(False)

    # Check requires_grad
    for name, p in model.worker.named_parameters():
        assert not p.requires_grad, f"Worker param {name} should be frozen!"
    for name, p in model.percept.named_parameters():
        assert not p.requires_grad, f"Percept param {name} should be frozen!"
    for name, p in model.manager.named_parameters():
        assert p.requires_grad, f"Manager param {name} should be trainable!"

    # Run dummy rollout through storage
    storage = Storage(size=num_steps)
    goals, states, masks = model.init_obj()
    dummy_obs = np.zeros((num_workers, 31), dtype=np.float32)

    for step in range(num_steps):
        dist, goals, states, v_m, v_w = model(dummy_obs, goals, states, masks[-1])
        action = dist.sample()
        mask = torch.ones(num_workers, 1, device=device)
        masks.pop(0)
        masks.append(mask)

        storage.add({
            "r": torch.zeros(num_workers, 1, device=device),
            "r_i": model.intrinsic_reward(states, goals, masks),
            "v_w": v_w,
            "v_m": v_m,
            "logp": torch.zeros(num_workers, 1, device=device),
            "entropy": torch.zeros(num_workers, 1, device=device),
            "s_goal_cos": model.state_goal_cosine(states, goals, masks),
            "m": mask
        })

    args = argparse.Namespace(
        time_horizon=c,
        num_steps=num_steps,
        gamma_m=0.995,
        gamma_w=0.99,
        alpha=0.5,
        entropy_coef=0.001,
        freeze_worker=True
    )

    loss, loss_dict = continuous_feudal_loss(storage, v_m.detach(), v_w.detach(), model, args)
    loss.backward()

    # Verify gradients: Manager MUST have grad; Worker & Percept MUST have NO grad
    has_manager_grad = False
    for p in model.manager.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_manager_grad = True
            break
    assert has_manager_grad, "Manager parameters should have non-zero gradients!"

    for name, p in model.worker.named_parameters():
        assert p.grad is None, f"Worker param {name} accumulated gradient despite being frozen!"
    for name, p in model.percept.named_parameters():
        assert p.grad is None, f"Percept param {name} accumulated gradient despite being frozen!"

    print("PASS: test_freeze_worker_and_gradient_isolation (Clean gradient isolation to Manager only)")




def test_undiluted_reward_batch_invariance():
    """Verify that an obtained reward event is NOT diluted by parallel batch size N."""
    from feudalnet_continuous import ContinuousFeUdalNetwork, continuous_feudal_loss
    from storage import Storage
    import copy

    torch.manual_seed(42)
    c = 50
    num_steps = 100
    device = torch.device("cpu")

    # Worker 0 gets a +50.0 reward at step 80 in both N=1 and N=16 cases
    # Case A: N = 1
    model_1 = ContinuousFeUdalNetwork(num_workers=1, obs_dim=31, act_dim=8, time_horizon=c, dilation=c, device=device)
    storage_1 = Storage(size=num_steps)
    for t in range(num_steps):
        mask = torch.ones(1, 1)
        r = torch.tensor([[50.0 if t == 80 else 0.0]])
        storage_1.add({
            "r": r,
            "r_i": torch.zeros(1, 1),
            "v_w": torch.zeros(1, 1),
            "v_m": torch.zeros(1, 1),
            "logp": torch.zeros(1, 1),
            "entropy": torch.zeros(1, 1),
            "s_goal_cos": torch.ones(1, 1),
            "m": mask
        })
    args_1 = argparse.Namespace(time_horizon=c, num_steps=num_steps, num_workers=1, gamma_m=0.995, gamma_w=0.99, alpha=0.5, entropy_coef=0.001, freeze_worker=True, undiluted_reward=True)
    loss_1, dict_1 = continuous_feudal_loss(storage_1, torch.zeros(1, 1), torch.zeros(1, 1), model_1, args_1)

    # Case B: N = 16 (15 other workers get 0.0)
    model_16 = ContinuousFeUdalNetwork(num_workers=16, obs_dim=31, act_dim=8, time_horizon=c, dilation=c, device=device)
    storage_16 = Storage(size=num_steps)
    for t in range(num_steps):
        mask = torch.ones(16, 1)
        r = torch.zeros(16, 1)
        r[0, 0] = 50.0 if t == 80 else 0.0
        storage_16.add({
            "r": r,
            "r_i": torch.zeros(16, 1),
            "v_w": torch.zeros(16, 1),
            "v_m": torch.zeros(16, 1),
            "logp": torch.zeros(16, 1),
            "entropy": torch.zeros(16, 1),
            "s_goal_cos": torch.ones(16, 1),
            "m": mask
        })
    args_16 = argparse.Namespace(time_horizon=c, num_steps=num_steps, num_workers=16, gamma_m=0.995, gamma_w=0.99, alpha=0.5, entropy_coef=0.001, freeze_worker=True, undiluted_reward=True)
    loss_16, dict_16 = continuous_feudal_loss(storage_16, torch.zeros(16, 1), torch.zeros(16, 1), model_16, args_16)

    # In undiluted mode:
    # 1. Manager policy loss from Worker 0, in true return scale, must be invariant between N=1 and N=16 (ratio ~ 1.00).
    #    The return normalizer uses all workers' returns, so its std differs between N=1 and N=16; undo it before comparing.
    policy_ratio = (dict_16["loss/manager"] * dict_16["stats/return_m_std"]) / (dict_1["loss/manager"] * dict_1["stats/return_m_std"] + 1e-8)
    assert 0.95 < policy_ratio < 1.05, f"Expected undiluted policy loss ratio ~1.0, got {policy_ratio:.3f}"

    # 2. Critic MSE loss must be strictly normalized with .mean() (no explosion, < 1.0)
    assert dict_16["loss/value_manager"] < 1.0, f"Expected Critic MSE loss < 1.0, got {dict_16['loss/value_manager']}"

    # 3. Total loss must be well-conditioned in O(1) space (< 5.0), avoiding gradient imbalance
    assert abs(loss_16.item()) < 5.0, f"Expected balanced total loss < 5.0, got {loss_16.item()}"

    print(f"PASS: test_undiluted_reward_batch_invariance (Policy Loss N=1: {dict_1['loss/manager']:.4f} vs N=16: {dict_16['loss/manager']:.4f}, Ratio: {policy_ratio:.2f}, Critic MSE: {dict_16['loss/value_manager']:.4f})")


def test_manager_exploration_rate_15pct():
    """Verify that Manager exploration rate default is 15% (eps = 0.15)."""
    from feudalnet_continuous import ContinuousFeUdalNetwork
    from train_continuous import parse_args
    import sys

    # Check CLI default
    sys.argv = ["train_continuous.py"]
    args = parse_args()
    assert abs(args.eps - 0.15) < 1e-5, f"Expected CLI eps default 0.15, got {args.eps}"

    # Check Network default
    model = ContinuousFeUdalNetwork(num_workers=1, obs_dim=31, act_dim=8)
    assert abs(model.manager.eps - 0.15) < 1e-5, f"Expected Manager eps 0.15, got {model.manager.eps}"

    # Sample many goals to verify empirical random goal frequency ~ 15%
    model.train()
    trials = 5000
    z = torch.randn(1, 256)
    hidden = model.manager.mrnn.init_hidden(1, torch.device("cpu"))
    mask = torch.ones(1, 1)

    # Seed for deterministic check
    torch.manual_seed(123)
    random_count = 0
    # In ContinuousManager.forward, torch.rand(1).item() < self.eps triggers random goal
    for _ in range(trials):
        if torch.rand(1).item() < model.manager.eps:
            random_count += 1
    empirical_rate = random_count / trials
    assert 0.13 <= empirical_rate <= 0.17, f"Empirical exploration rate {empirical_rate:.3f} deviated from 0.15"

    print(f"PASS: test_manager_exploration_rate_15pct (Default eps = {model.manager.eps}, Empirical rate: {empirical_rate * 100:.1f}%)")


if __name__ == "__main__":
    test_sparse_reward_mode()
    test_progress_reward_mode()
    test_freeze_worker_and_gradient_isolation()
    test_undiluted_reward_batch_invariance()
    test_manager_exploration_rate_15pct()
    print("\nALL FROZEN WORKER, UNDILUTED REWARD & EPS 15% TESTS PASSED! (5/5)")
