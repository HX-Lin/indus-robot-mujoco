import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp

from vita_jax.config import ModelConfig
from vita_jax.ode import VITAODEHead
from vita_jax.ppf import PPFFutureVisualHead, ppf_future_visual_loss


def test_ppf_future_target_is_detached_and_masked():
    head = PPFFutureVisualHead(action_cond_dim=4 * 14, visual_dim=8,
                               hidden_dim=16)
    key = jax.random.PRNGKey(0)
    pred = jax.random.normal(key, (2, 6, 14))
    current = jax.random.normal(jax.random.PRNGKey(1), (2, 8))
    future = jax.random.normal(jax.random.PRNGKey(2), (2, 8))
    variables = head.init(key, pred[:, :4].reshape(2, -1), current)
    valid = jnp.asarray([True, False])

    def apply_head(action, cur):
        return head.apply(variables, action, cur)

    def loss_for(fut, cur):
        return ppf_future_visual_loss(apply_head, pred, cur, fut, valid,
                                      horizon=4)[0]

    loss = loss_for(future, current)
    # The future target is stop-gradiented.
    g_future = jax.grad(lambda f: loss_for(f, current))(future)
    assert jnp.allclose(g_future, 0.0)
    # The current visual input is attached to the head.
    g_current = jax.grad(lambda c: loss_for(future, c))(current)
    assert jnp.any(jnp.abs(g_current) > 0)

    # Corrupting the masked-out future row must not change the loss.
    bad = future.at[1].set(1e6)
    assert jnp.allclose(loss_for(bad, current), loss)


def test_ppf_gradient_reaches_ode_parameters():
    cfg = ModelConfig(ode_hidden_dim=16, ode_num_layers=2, chunk_size=6,
                      ppf_future_visual_hidden=16)
    ode = VITAODEHead(cfg, pooled_dim=8)
    head = PPFFutureVisualHead(action_cond_dim=4 * 14, visual_dim=8,
                               hidden_dim=16)
    cond = {"pooled": jnp.zeros((2, 8)),
            "patches": jnp.zeros((2, 2, 3, 512, 2, 2)),
            "weights": jnp.ones((2, 2, 3, 1, 2, 2))}
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 2, 16))
    sm, ss, pm, ps = (jnp.zeros((16,)), jnp.ones((16,)),
                      jnp.zeros((14,)), jnp.ones((14,)))
    ode_vars = ode.init(jax.random.PRNGKey(1), cond, obs, sm, ss, pm, ps,
                        train=False)
    current = jax.random.normal(jax.random.PRNGKey(2), (2, 8))
    future = jax.random.normal(jax.random.PRNGKey(3), (2, 8))
    ppf_vars = head.init(jax.random.PRNGKey(4),
                         jnp.zeros((2, 4 * 14)), current)
    valid = jnp.ones((2,))

    def loss(params):
        pred = ode.apply({"params": params}, cond, obs, sm, ss, pm, ps,
                         train=False)
        return ppf_future_visual_loss(
            lambda a, c: head.apply(ppf_vars, a, c),
            pred, current, future, valid, horizon=4)[0]

    grad = jax.grad(loss)(ode_vars["params"])
    total = sum(float(jnp.sum(jnp.abs(x)))
                for x in jax.tree_util.tree_leaves(grad))
    assert total > 0
