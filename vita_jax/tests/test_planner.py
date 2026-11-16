import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp
import numpy as np

from vita_jax.config import (DataConfig, ModelConfig, TextConfig, TrainConfig,
                             VITAConfig)
from vita_jax.policy import VITAPolicy
from vita_jax.text import (SyntheticTextEncoder, TextBank,
                           TransformersFlaxT5Encoder, plan_for, serialize,
                           t5_teacher_forced_loss, teacher_forced_decoder_inputs)


TASK = "Stack red block, green block, and blue block centrally with blue block on top."


def test_text_bank_uses_literal_generated_plan_fields():
    cfg = TextConfig(encoder="synthetic", joint_planner=True,
                     generated_text_targets=True, task_instruction=TASK,
                     order="blue red green", max_length=16,
                     planner_max_length=32, planner_labels_max_length=24)
    bank = TextBank(cfg)
    plan = plan_for(("blue block", "red block", "green block"))
    assert bank.specs.clauses == [serialize([row]) for row in plan]
    assert bank.specs.target_descriptions == [
        "blue block", "red block", "green block"]
    assert bank.specs.reference_texts == [
        "center", "on blue block", "on red block"]

    batch = bank.planning_batch(offset=0, batch_size=2)
    assert batch["labels"].shape[0] == 3
    assert np.array_equal(np.asarray(batch["labels"][-1]),
                          bank.planning_task["labels"][0])
    # Auxiliary rows are deterministic and followed by the paired task row.
    assert np.array_equal(np.asarray(bank.planning_batch(0, 2)["labels"]),
                          np.asarray(batch["labels"]))

    raw = ("pick_place | foundation red block | tray ; "
           "pick_place | mug | beside box ; "
           "pick_place | spare tool | work area")
    bank.set_generated_plan(raw)
    assert bank.specs.target_descriptions == [
        "foundation red block", "mug", "spare tool"]
    assert bank.specs.reference_texts == ["tray", "beside box", "work area"]
    assert bank.specs.clauses[1] == "pick_place | mug | beside box"


def test_teacher_forced_t5_loss_is_differentiable_and_ignores_pad():
    from transformers import FlaxT5ForConditionalGeneration, T5Config

    config = T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32,
                      num_layers=1, num_decoder_layers=1, num_heads=2,
                      pad_token_id=0, eos_token_id=1,
                      decoder_start_token_id=0)
    model = FlaxT5ForConditionalGeneration(config)
    params = model.init_weights(jax.random.PRNGKey(0), (2, 8))

    adapter = TransformersFlaxT5Encoder.__new__(TransformersFlaxT5Encoder)
    adapter.model = model
    adapter.joint_planner = True
    adapter.pad_token_id = 0
    adapter.output_dim = 16

    labels = jnp.asarray([[7, 8, 9, 1, -100, -100],
                          [10, 11, 1, -100, -100, -100]], jnp.int32)
    batch = {
        "input_ids": jnp.asarray([[5, 6, 1, 0, 0, 0],
                                  [12, 13, 14, 1, 0, 0]], jnp.int32),
        "attention_mask": jnp.asarray([[1, 1, 1, 0, 0, 0],
                                       [1, 1, 1, 1, 0, 0]], jnp.int32),
        "labels": labels,
    }
    loss = adapter.planning_loss(params, batch, train=False)
    assert jnp.isfinite(loss)

    target_ids, decoder_ids = teacher_forced_decoder_inputs(labels, 0, 0)
    assert jnp.array_equal(target_ids, jnp.asarray(
        [[7, 8, 9, 1, 0, 0], [10, 11, 1, 0, 0, 0]]))
    assert jnp.array_equal(decoder_ids[:, 0], jnp.zeros((2,), jnp.int32))
    assert jnp.array_equal(decoder_ids[:, 1:], target_ids[:, :-1])

    grads = jax.grad(lambda p: adapter.planning_loss(p, batch, train=False))(
        params)
    for group in ("shared", "encoder", "decoder"):
        total = sum(float(jnp.sum(jnp.abs(x)))
                    for x in jax.tree_util.tree_leaves(grads[group]))
        assert total > 0, group
    if "lm_head" in grads:
        total = sum(float(jnp.sum(jnp.abs(x)))
                    for x in jax.tree_util.tree_leaves(grads["lm_head"]))
        assert total > 0

    # The CE mean is over valid labels only; pad positions never contribute.
    _, decoder_ids = teacher_forced_decoder_inputs(labels, 0, 0)
    out = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=decoder_ids,
                decoder_attention_mask=jnp.ones_like(decoder_ids),
                train=False, params=params)
    log_probs = jax.nn.log_softmax(out.logits, axis=-1)
    picked = jnp.take_along_axis(
        log_probs, target_ids[..., None], axis=-1)[..., 0]
    valid = labels >= 0
    expected = -jnp.sum(jnp.where(valid, picked, 0.0)) / jnp.sum(valid)
    assert jnp.allclose(loss, expected)


class _FakePlanner(SyntheticTextEncoder):
    joint_planner = True

    def planning_loss(self, params, batch, train=False, dropout_rng=None):
        # Constant non-zero loss makes the total-objective weighting explicit.
        return jnp.asarray(1.25)


def test_policy_total_objective_adds_joint_plan_weight():
    cfg = VITAConfig(
        model=ModelConfig(ppf_future_visual_weight=0.0,
                          joint_plan_weight=0.5, ode_hidden_dim=16,
                          ode_num_layers=2, resize_shape=(32, 32),
                          crop_shape=(24, 24), state_dropout=0.0,
                          ode_perturb_prob=0.0),
        text=TextConfig(encoder="synthetic", joint_planner=True,
                        generated_text_targets=True, task_instruction=TASK,
                        max_length=8, joint_plan_batch_size=2,
                        planner_max_length=32, planner_labels_max_length=16),
        data=DataConfig(data_dir="x", repo_id="x", ee_cache="x"),
        train=TrainConfig(steps=1, batch_size=2))
    policy = VITAPolicy.create(cfg, text_encoder=_FakePlanner(output_dim=8))
    params, constants, batch_stats = policy.init(
        jax.random.PRNGKey(0), batch_size=2, height=32, width=32)
    batch = policy.dummy_batch(2, 32, 32)
    base = policy.apply(params, constants, batch_stats, batch, train=False)
    total, (metrics, _) = policy.loss(
        params, constants, batch_stats, batch, train=False, planner_index=0)
    assert jnp.allclose(metrics["joint_plan_loss"], 1.25)
    assert jnp.allclose(total, base["loss"] + 0.5 * 1.25)
    assert jnp.allclose(metrics["total_loss"], total)

    train_total, (train_metrics, _) = policy.loss(
        params, constants, batch_stats, batch, train=True,
        rng=jax.random.PRNGKey(1), planner_index=1)
    assert jnp.allclose(train_metrics["joint_plan_weighted_loss"], 0.625)
    assert jnp.isfinite(train_total)
