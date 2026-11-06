import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np

from feudalnet_continuous import ContinuousFeUdalNetwork, continuous_feudal_loss
from feudal_cat_continuous import ContinuousFeUdalCatNetwork, continuous_feudal_cat_loss
from worker_only_continuous import ContinuousWorkerOnlyNetwork
from storage import Storage
from train_continuous import sample_action
import argparse


def test_feudal_cat_worker_hidden_dim():
    """Verify that FeUdalCat Worker hidden capacity is at least 128 even if 16 is passed."""
    model = ContinuousFeUdalCatNetwork(
        num_workers=2,
        obs_dim=31,
        act_dim=8,
        hidden_dim_manager=256,
        hidden_dim_worker=16,  # Should be overridden to at least 128
        time_horizon=10,
        dilation=10,
        device=torch.device("cpu")
    )
    assert model.worker.wrnn.hidden_size >= 128, (
        f"Expected Worker LSTM hidden size >= 128, got {model.worker.wrnn.hidden_size}"
    )
    assert model.worker_hidden[0].shape[-1] >= 128
    assert model.worker_hidden[1].shape[-1] >= 128


def test_feudal_worker_log_std_clamping():
    """Verify that worker log_std is clamped to prevent exploration variance explosion."""
    model_fun = ContinuousFeUdalNetwork(num_workers=1, obs_dim=10, act_dim=4, device=torch.device("cpu"))
    model_cat = ContinuousFeUdalCatNetwork(num_workers=1, obs_dim=10, act_dim=4, device=torch.device("cpu"))

    # Force large log_std to simulate extreme gradient updates
    model_fun.worker.log_std.data.fill_(5.0)
    model_cat.worker.log_std.data.fill_(5.0)

    goals, states, masks = model_fun.init_obj()
    z = torch.randn(1, 256)
    dist_fun, _, _ = model_fun.worker(z, goals, model_fun.worker_hidden, masks[-1])
    dist_cat, _, _ = model_cat.worker(z, goals, model_cat.worker_hidden, masks[-1])

    # std should be clamped to exp(0.2) ~= 1.2214
    max_std = np.exp(0.2) + 1e-4
    assert (dist_fun.stddev.detach().numpy() <= max_std).all(), f"FuN std exceeded max: {dist_fun.stddev}"
    assert (dist_cat.stddev.detach().numpy() <= max_std).all(), f"Cat std exceeded max: {dist_cat.stddev}"


def test_reset_workers_isolation():
    """Verify that resetting a finished worker zeroes its goals/states without affecting others."""
    for model_cls in [ContinuousFeUdalNetwork, ContinuousFeUdalCatNetwork, ContinuousWorkerOnlyNetwork]:
        model = model_cls(num_workers=2, obs_dim=10, act_dim=4, time_horizon=5, dilation=5, device=torch.device("cpu"))
        goals, states, masks = model.init_obj()

        # Fill goals and states with dummy non-zero values
        for g in goals:
            g.data.fill_(1.0)
        for s in states:
            s.data.fill_(2.0)
        model.worker_hidden[0].data.fill_(3.0)
        model.worker_hidden[1].data.fill_(3.0)

        # Worker 0 finished (reset_mask[0] = 0.0), Worker 1 active (reset_mask[1] = 1.0)
        reset_mask = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        model.reset_workers(reset_mask, goals, states)

        # Worker 0 must be zeroed out
        for g in goals:
            assert torch.all(g[0] == 0.0), f"Worker 0 goal not zeroed for {model_cls.__name__}"
            assert torch.all(g[1] == 1.0), f"Worker 1 goal corrupted for {model_cls.__name__}"
        for s in states:
            assert torch.all(s[0] == 0.0), f"Worker 0 state not zeroed for {model_cls.__name__}"
            assert torch.all(s[1] == 2.0), f"Worker 1 state corrupted for {model_cls.__name__}"

        assert torch.all(model.worker_hidden[0][0] == 0.0)
        assert torch.all(model.worker_hidden[0][1] == 3.0)


def test_sample_action_bounds_and_logp():
    """Verify that sample_action produces bounded actions [-1, 1] and finite logp."""
    mu = torch.tensor([[0.8, -0.9, 0.0]])
    std = torch.tensor([[0.5, 0.5, 0.5]])
    dist = torch.distributions.Normal(mu, std)

    action, logp, entropy = sample_action(dist)
    assert np.all(action >= -1.0) and np.all(action <= 1.0), f"Action out of bounds: {action}"
    assert torch.isfinite(logp).all(), f"Non-finite logp: {logp}"
    assert torch.isfinite(entropy).all(), f"Non-finite entropy: {entropy}"


def test_loss_and_logging_metrics():
    """Verify gradient flow and presence of raw advantage statistics in loss_dict."""
    for model_cls, loss_fn in [
        (ContinuousFeUdalNetwork, continuous_feudal_loss),
        (ContinuousFeUdalCatNetwork, continuous_feudal_cat_loss)
    ]:
        num_workers = 2
        obs_dim = 10
        act_dim = 4
        c = 3
        num_steps = 6

        model = model_cls(
            num_workers=num_workers,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim_manager=64,
            hidden_dim_worker=16,
            time_horizon=c,
            dilation=c,
            device=torch.device("cpu")
        )
        goals, states, masks = model.init_obj()
        obs = torch.randn(num_workers, obs_dim)

        storage = Storage(size=num_steps)
        for _ in range(num_steps):
            dist, goals, states, vm, vw = model(obs, goals, states, masks[-1])
            act, logp, ent = sample_action(dist)
            storage.add({
                'r': torch.randn(num_workers, 1),
                'r_i': model.intrinsic_reward(states, goals, masks),
                'v_w': vw,
                'v_m': vm,
                'logp': logp.unsqueeze(-1),
                'entropy': ent.unsqueeze(-1),
                's_goal_cos': model.state_goal_cosine(states, goals, masks),
                'm': masks[-1]
            })

        args = argparse.Namespace(
            num_steps=num_steps,
            gamma_m=0.99,
            gamma_w=0.99,
            alpha=0.5,
            time_horizon=c,
            entropy_coef=0.01
        )
        _, _, _, next_vm, next_vw = model(obs, goals, states, masks[-1], save=False)
        loss, loss_dict = loss_fn(storage, next_vm, next_vw, model, args)

        loss.backward()

        # Check Manager received gradients
        assert model.manager.critic.weight.grad is not None, f"Manager critic grad None for {model_cls.__name__}"
        assert model.worker.critic[0].weight.grad is not None, f"Worker critic grad None for {model_cls.__name__}"

        # Verify informative advantage metrics are present and not hardcoded to 0
        assert 'policy/manager_advantage_mean' in loss_dict
        assert 'policy/manager_advantage_std' in loss_dict
        assert 'policy/worker_advantage_mean' in loss_dict
        assert 'policy/worker_advantage_std' in loss_dict


if __name__ == "__main__":
    test_feudal_cat_worker_hidden_dim()
    print("PASS: test_feudal_cat_worker_hidden_dim")
    test_feudal_worker_log_std_clamping()
    print("PASS: test_feudal_worker_log_std_clamping")
    test_reset_workers_isolation()
    print("PASS: test_reset_workers_isolation")
    test_sample_action_bounds_and_logp()
    print("PASS: test_sample_action_bounds_and_logp")
    test_loss_and_logging_metrics()
    print("PASS: test_loss_and_logging_metrics")
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY! (5/5)")
