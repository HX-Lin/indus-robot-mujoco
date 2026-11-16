import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from vita_jax.policy import assemble_objective


def test_train_adds_ppf_eval_does_not():
    mse = jnp.asarray(0.3)
    l1 = jnp.asarray(0.2)
    ground = jnp.asarray(0.4)
    route = jnp.asarray(0.5)
    ppf = jnp.asarray(0.7)
    weight = 0.1
    train_total, train_base = assemble_objective(
        mse, l1, ground, route, ppf, weight, True,
        flow_recon_weight=0.5, grounding_weight=0.5, stage_route_weight=0.5)
    eval_total, eval_base = assemble_objective(
        mse, l1, ground, route, ppf, weight, False,
        flow_recon_weight=0.5, grounding_weight=0.5, stage_route_weight=0.5)
    expected_base = 0.3 + 0.5 * 0.2 + 0.5 * 0.4 + 0.5 * 0.5
    assert jnp.allclose(train_base, expected_base)
    assert jnp.allclose(eval_base, expected_base)
    assert jnp.allclose(train_total, expected_base + weight * 0.7)
    assert jnp.allclose(eval_total, expected_base)


def test_policy_train_eval_objective_accounting():
    pytest.importorskip("flax")
    import jax
    from vita_jax.config import (DataConfig, ModelConfig, TextConfig,
                                 TrainConfig, VITAConfig)
    from vita_jax.policy import VITAPolicy

    cfg = VITAConfig(
        model=ModelConfig(ppf_future_visual_weight=0.1, ode_hidden_dim=16,
                          ode_num_layers=2, ppf_future_visual_hidden=16,
                          resize_shape=(32, 32), crop_shape=(24, 24),
                          state_dropout=0.0, ode_perturb_prob=0.0),
        text=TextConfig(encoder="synthetic", max_length=8),
        data=DataConfig(data_dir="x", repo_id="x", ee_cache="x"),
        train=TrainConfig(steps=1, batch_size=2))
    policy = VITAPolicy.create(cfg)
    params, constants, batch_stats = policy.init(
        jax.random.PRNGKey(0), batch_size=2, height=32, width=32)
    batch = policy.dummy_batch(2, 32, 32)
    eval_out = policy.apply(params, constants, batch_stats, batch, train=False)
    train_out, _ = policy.apply(
        params, constants, batch_stats, batch, train=True,
        mutable=("batch_stats",), rng=jax.random.PRNGKey(1))
    assert jnp.allclose(
        eval_out["metrics"]["total_loss"], eval_out["metrics"]["base_loss"])
    assert jnp.allclose(
        train_out["metrics"]["total_loss"],
        train_out["metrics"]["base_loss"]
        + cfg.model.ppf_future_visual_weight * train_out["ppf_loss"])
