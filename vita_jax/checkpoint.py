"""Small Flax checkpoint helper (params, constants, batch stats, optimizer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Tuple

import flax.serialization
import jax
import numpy as np


def save_checkpoint(path, params, constants, batch_stats, opt_state, step: int,
                    metadata: Mapping[str, Any]):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "params": params,
        "constants": constants,
        "batch_stats": batch_stats,
    }
    (path / "state.msgpack").write_bytes(flax.serialization.to_bytes(state))
    (path / "optimizer.msgpack").write_bytes(flax.serialization.to_bytes(opt_state))
    def jsonable(value):
        if isinstance(value, (np.ndarray, jax.Array)):
            return np.asarray(value).tolist()
        if isinstance(value, dict):
            return {k: jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(v) for v in value]
        return value
    (path / "metadata.json").write_text(
        json.dumps({"step": int(step), **jsonable(dict(metadata))}, indent=2))


def load_checkpoint(path, params_template, constants_template,
                    batch_stats_template, opt_state_template
                    ) -> Tuple[Mapping, Mapping, Mapping, Any, int, dict]:
    path = Path(path)
    state = flax.serialization.from_bytes(
        {"params": params_template, "constants": constants_template,
         "batch_stats": batch_stats_template},
        (path / "state.msgpack").read_bytes())
    opt_state = flax.serialization.from_bytes(
        opt_state_template, (path / "optimizer.msgpack").read_bytes())
    meta = json.loads((path / "metadata.json").read_text())
    return (state["params"], state["constants"], state["batch_stats"],
            opt_state, int(meta.get("step", 0)), meta)
