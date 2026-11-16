"""Trainable Flan-T5 text conditioning and joint planner for the JAX port.

``TransformersFlaxT5Encoder`` is the production adapter.  With a joint planner
it loads the local Flan-T5-small seq2seq checkpoint through ``transformers``'
Flax T5 implementation (``from_pt=True`` when only PyTorch weights are present),
shares the encoder/shared parameters with FASA, and exposes a differentiable
teacher-forced decoder loss.  Without a planner it loads the encoder subtree
only, which is useful for evaluation or explicit ablations.

The planner data mirrors ``cs-platform/flan_subtask_planner.py``: the strict
``PROMPT``/``serialize`` grammar, the deterministic template examples, and the
active robot instruction with its demonstrated-order gold plan.  Auxiliary
text samples never relabel the robot demonstration and only enter the language
loss.

If the Flax T5 stack is unavailable, ``UnavailableTextEncoder`` fails loudly
instead of silently substituting a frozen cache.  ``SyntheticTextEncoder`` is a
deterministic test-only token encoder; it does not claim to reproduce T5.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import TextConfig


PLAN_PROMPT = (
    "Generate robot subtasks from the instruction. Each subtask must be: "
    "pick_place | object description | destination description. Separate subtasks with semicolons. "
    "Use center for the first destination and on <object description> for stacking. "
    "Output only the ordered subtasks. Instruction: "
)
GENERIC_PLAN_PROMPT = (
    "Generate robot subtasks from the instruction. Each subtask must be: "
    "skill | object description | destination description. Separate subtasks with semicolons. "
    "Output only the ordered subtasks. Instruction: "
)
TRAIN_TEMPLATES = [
    "Stack {a} at the bottom, {b} in the middle, and {c} on top.",
    "Put {a} in the center. Put {b} on {a}. Put {c} on {b}.",
    "Build a tower with {c} at the top, {b} in the middle, and {a} at the bottom.",
    "{b} must rest on {a}, and {c} must rest on {b}. Build this stack centrally.",
    "Make {a} the base and {c} the top. Place {b} between them.",
    "From lowest to highest, arrange {a}, {b}, {c} in a stack.",
    "From highest to lowest the tower is {c}, {b}, {a}. Assemble it.",
    "The objects are {mention}. Ignore that list order. The stack needs {a} below {b} below {c}.",
    "First place {a} centrally, next stack {b} on it, and finally add {c}.",
    "{c} belongs above {b}. {a} belongs below {b}. Stack these objects.",
]
TEST_TEMPLATES = {
    "balanced": ["Create a central pile: foundation {a}; second layer {b}; final layer {c}."],
    "reverse": ["Reading downward, the requested tower has {c} over {b} over {a}."],
    "distractor": ["Inventory: {mention}. The inventory is not a sequence. "
                   "Keep {b} between {a} underneath and {c} above."],
    "heldout_phrasing": [
        "Start the tower with {a}, cap it using {c}, and fit {b} between those two.",
        "Only {a} touches the table. {b} supports {c} and is supported by {a}.",
    ],
}
OBJECT_GROUPS = [
    ("red block", "green block", "blue block"),
    ("yellow cube", "purple cube", "orange cube"),
    ("small box", "medium box", "large box"),
    ("wooden block", "metal block", "plastic block"),
]


def plan_for(order: Sequence[str]) -> List[dict]:
    a, b, c = order
    return [dict(skill="pick_place", object=a, destination="center"),
            dict(skill="pick_place", object=b, destination="on " + a),
            dict(skill="pick_place", object=c, destination="on " + b)]


def serialize(plan: Sequence[Mapping[str, str]]) -> str:
    return " ; ".join(
        f"{p['skill']} | {p['object']} | {p['destination']}" for p in plan)


def parse_plan(text: str) -> List[dict]:
    """Validate syntax only; never infer, deduplicate or replace model output."""
    lines = str(text).strip().split(";")
    if len(lines) != 3:
        raise ValueError("expected three generated subtasks")
    result = []
    for line in lines:
        fields = [part.strip() for part in line.split("|")]
        if len(fields) != 3 or fields[0] != "pick_place" or not all(fields):
            raise ValueError("invalid generated subtask fields")
        result.append(dict(skill=fields[0], object=fields[1],
                           destination=fields[2]))
    return result


def examples() -> Tuple[List[dict], Dict[str, List[dict]]]:
    """Deterministic planning examples copied from the PyTorch planner data."""
    train = []
    for group in OBJECT_GROUPS:
        for order in itertools.permutations(group):
            for template in TRAIN_TEMPLATES:
                mentions = (list(itertools.permutations(group))
                            if "{mention}" in template else [group])
                for mention in mentions:
                    instruction = template.format(
                        a=order[0], b=order[1], c=order[2],
                        mention=", ".join(mention))
                    train.append(dict(instruction=instruction,
                                      gold=plan_for(order)))
    tests = {}
    for split, templates in TEST_TEMPLATES.items():
        tests[split] = []
        for order in itertools.permutations(OBJECT_GROUPS[0]):
            for template in templates:
                mention = (order[2], order[0], order[1])
                tests[split].append(dict(
                    instruction=template.format(
                        a=order[0], b=order[1], c=order[2],
                        mention=", ".join(mention)),
                    gold=plan_for(order)))
    tests["heldout_objects"] = []
    for order in itertools.permutations(
            ("cyan cylinder", "pink cylinder", "brown cylinder")):
        tests["heldout_objects"].append(dict(
            instruction=TRAIN_TEMPLATES[1].format(
                a=order[0], b=order[1], c=order[2]),
            gold=plan_for(order)))
    return train, tests


def planning_examples(seed: int = 42) -> List[dict]:
    rows, _ = examples()
    random.Random(seed).shuffle(rows)
    return rows


@dataclass
class TextSpecs:
    order: Tuple[str, str, str]
    clauses: List[str]
    target_descriptions: List[str]
    reference_texts: List[str]

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)


def _generated_text_specs(config: TextConfig
                          ) -> Optional[Tuple[List[str], List[str], List[str]]]:
    if not config.generated_text_targets:
        return None
    if not config.joint_planner:
        raise ValueError("generated text targets require a generative planner")
    order = tuple(config.order.split())
    plan = plan_for(tuple(colour + " block" for colour in order))
    descriptions = list(config.target_descriptions or
                        [item["object"] for item in plan])
    references = list(config.reference_texts or
                      [item["destination"] for item in plan])
    if len(descriptions) != len(plan):
        raise ValueError("target_descriptions must align with the plan")
    if len(references) != len(plan):
        raise ValueError("reference_texts must align with the plan")
    clauses = [
        f"{item['skill']} | {descriptions[index]} | {item['destination']}"
        for index, item in enumerate(plan)
    ]
    return clauses, descriptions, references


def build_text_specs(config: TextConfig) -> TextSpecs:
    order = tuple(config.order.split())
    if len(order) != 3 or len(set(order)) != 3:
        raise ValueError("language order must contain three distinct colours")
    generated = _generated_text_specs(config)
    if generated is not None:
        clauses, descriptions, references = generated
    else:
        b, m, t = order
        clauses = [
            f"{b} block at the center",
            f"{m} block on the {b} block",
            f"{t} block on the {m} block",
        ]
        descriptions = list(config.target_descriptions or
                            [c + " block" for c in order])
        references = (list(config.reference_texts)
                      if config.reference_texts is not None
                      else ["the center", f"the {b} block", f"the {m} block"])
    if len(descriptions) != len(clauses):
        raise ValueError("target_descriptions must align with the clause count")
    if len(references) != len(clauses):
        raise ValueError("reference_texts must align with the clause count")
    return TextSpecs(order=order, clauses=clauses,
                     target_descriptions=descriptions,
                     reference_texts=references)


class _NumpyTokenizer:
    """Tiny deterministic tokenizer for synthetic tests only."""

    pad_token_id = 0

    def __call__(self, texts, padding="max_length", truncation=True, max_length=16,
                 return_tensors="np"):
        if isinstance(texts, str):
            texts = [texts]
        ids = np.zeros((len(texts), max_length), dtype=np.int32)
        mask = np.zeros((len(texts), max_length), dtype=np.int32)
        for row, text in enumerate(texts):
            words = str(text).lower().replace("|", " ").split()
            for col, word in enumerate(words[:max_length]):
                value = 1
                for char in word:
                    value = (value * 131 + ord(char)) % 32000
                ids[row, col] = 1 + value
                mask[row, col] = 1
            if not words:
                ids[row, 0] = 1
                mask[row, 0] = 1
        return {"input_ids": ids, "attention_mask": mask}


@dataclass
class TextArrays:
    clauses: Dict[str, np.ndarray]
    objects: Dict[str, np.ndarray]
    references: Dict[str, np.ndarray]

    def as_jax(self):
        import jax.numpy as jnp

        def convert(d):
            return {k: jnp.asarray(v) for k, v in d.items()}

        return {"clauses": convert(self.clauses),
                "objects": convert(self.objects),
                "references": convert(self.references)}


class TextBank:
    """Fixed task text plus deterministic planner teacher-forcing batches."""

    def __init__(self, config: TextConfig, tokenizer=None):
        self.config = config
        self.specs = build_text_specs(config)
        self.tokenizer = tokenizer or _NumpyTokenizer()
        self.max_length = int(config.max_length)
        self.pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        self.arrays = TextArrays(
            clauses=self._tokenize(self.specs.clauses),
            objects=self._tokenize(self.specs.target_descriptions),
            references=self._tokenize(self.specs.reference_texts),
        )
        self.planning_aux = None
        self.planning_task = None
        if config.joint_planner:
            self._build_planning()

    def _tokenize(self, texts: Sequence[str], max_length: Optional[int] = None
                  ) -> Dict[str, np.ndarray]:
        enc = self.tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=int(max_length or self.max_length), return_tensors="np")
        return {
            "input_ids": np.asarray(enc["input_ids"], dtype=np.int32),
            "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int32),
        }

    def _planning_arrays(self, instructions: Sequence[str],
                         golds: Sequence[Sequence[Mapping[str, str]]]):
        inputs = self._tokenize(
            [PLAN_PROMPT + text for text in instructions],
            max_length=self.config.planner_max_length)
        labels = self._tokenize(
            [serialize(gold) for gold in golds],
            max_length=self.config.planner_labels_max_length)
        target_ids = labels["input_ids"].copy()
        target_ids[labels["attention_mask"] == 0] = -100
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": target_ids.astype(np.int32),
        }

    def _build_planning(self):
        if not self.config.task_instruction:
            raise ValueError(
                "the joint T5 planner requires the original task instruction")
        rows = planning_examples()
        self.planning_aux = self._planning_arrays(
            [row["instruction"] for row in rows],
            [row["gold"] for row in rows])
        demonstrated = plan_for(tuple(colour + " block"
                                      for colour in self.specs.order))
        self.planning_task = self._planning_arrays(
            [self.config.task_instruction], [demonstrated])

    def jax_arrays(self):
        return self.arrays.as_jax()

    def planning_batch(self, offset=0, batch_size: int = 8):
        """Auxiliary cyclic rows followed by the paired active robot sample."""
        if self.planning_task is None:
            raise ValueError("planning batches require a joint planner")
        import jax.numpy as jnp

        batch_size = int(batch_size)
        if batch_size > 0:
            count = self.planning_aux["labels"].shape[0]
            indices = jnp.mod(jnp.asarray(offset) + jnp.arange(batch_size),
                              count)
            batch = {key: jnp.take(jnp.asarray(value), indices, axis=0)
                     for key, value in self.planning_aux.items()}
            for key, value in self.planning_task.items():
                batch[key] = jnp.concatenate(
                    [batch[key], jnp.asarray(value)], axis=0)
            return batch
        return {key: jnp.asarray(value)
                for key, value in self.planning_task.items()}

    def set_generated_plan(self, raw: str):
        """Use literal generated plan fields, matching the PyTorch contract."""
        if not self.config.generated_text_targets:
            raise ValueError(
                "set_generated_plan requires generated_text_targets so no "
                "semantic colour/stack repair is applied")
        plan = parse_plan(raw)
        self.generated_plan = plan
        self.generated_plan_text = str(raw)
        self.specs = TextSpecs(
            order=self.specs.order,
            clauses=[serialize([item]) for item in plan],
            target_descriptions=[item["object"] for item in plan],
            reference_texts=[item["destination"] for item in plan],
        )
        self.arrays = TextArrays(
            clauses=self._tokenize(self.specs.clauses),
            objects=self._tokenize(self.specs.target_descriptions),
            references=self._tokenize(self.specs.reference_texts),
        )
        return plan


def teacher_forced_decoder_inputs(labels, pad_token_id: int,
                                  decoder_start_token_id: int):
    """Shift right like Hugging Face T5 when only labels are supplied."""
    import jax.numpy as jnp

    labels = jnp.asarray(labels)
    target_ids = jnp.where(labels < 0, int(pad_token_id), labels)
    start = jnp.full((labels.shape[0], 1), int(decoder_start_token_id),
                     dtype=target_ids.dtype)
    decoder_input_ids = jnp.concatenate(
        [start, target_ids[:, :-1]], axis=1)
    return target_ids, decoder_input_ids


def t5_teacher_forced_loss(logits, labels):
    """Cross-entropy over shifted labels with pad (-100) ignored."""
    import jax
    import jax.numpy as jnp

    labels = jnp.asarray(labels)
    logits = jnp.asarray(logits)
    valid = labels >= 0
    targets = jnp.clip(labels, 0, logits.shape[-1] - 1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    total = jnp.sum(jnp.where(valid, picked, 0.0))
    return -total / jnp.maximum(jnp.sum(valid), 1)


class TextEncoder:
    """Interface for the separate, trainable text-parameter subtree."""

    output_dim: int
    trainable: bool = True
    joint_planner: bool = False

    def init_params(self, rng) -> Mapping:
        return {}

    def encode(self, params, ids, mask):
        raise NotImplementedError


class SyntheticTextEncoder(TextEncoder):
    """Deterministic test-only encoder; no pretrained semantics."""

    def __init__(self, output_dim: int = 32, trainable: bool = False,
                 frequency: float = 0.17):
        self.output_dim = int(output_dim)
        self.trainable = bool(trainable)
        self.frequency = float(frequency)

    def encode(self, params, ids, mask):
        import jax.numpy as jnp

        ids = jnp.asarray(ids)
        mask = jnp.asarray(mask)
        pos = jnp.arange(ids.shape[1], dtype=jnp.float32)[None, :, None]
        dim = jnp.arange(self.output_dim, dtype=jnp.float32)[None, None, :]
        value = jnp.sin(ids[..., None].astype(jnp.float32) * self.frequency
                        + pos * 0.31 + dim * 0.07)
        return value * mask[..., None].astype(value.dtype)


class UnavailableTextEncoder(TextEncoder):
    def __init__(self, reason: str):
        self.reason = reason
        self.output_dim = 0

    def encode(self, params, ids, mask):
        raise RuntimeError(self.reason)


class TransformersFlaxT5Encoder(TextEncoder):
    """Local Flan-T5-small encoder, optionally with the shared seq2seq decoder."""

    def __init__(self, model_path: str, max_length: int = 16,
                 joint_planner: bool = False, planner_max_length: int = 192,
                 planner_labels_max_length: int = 128):
        self.model_path = str(model_path)
        self.max_length = int(max_length)
        self.joint_planner = bool(joint_planner)
        self.planner_max_length = int(planner_max_length)
        self.planner_labels_max_length = int(planner_labels_max_length)
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency specific
            raise ImportError(
                "transformers is required for the Flax T5 text adapter") from exc
        try:
            import flax.linen as nn  # noqa: F401
            import jax.numpy as jnp  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError("flax and jax are required for the Flax T5 adapter") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True)
        self.pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        self.model, self.params = self._load_model()
        self.output_dim = int(self.model.config.d_model)

    def _load_model(self):
        from transformers import FlaxT5EncoderModel, FlaxT5ForConditionalGeneration
        errors = []
        model_cls = (FlaxT5ForConditionalGeneration if self.joint_planner
                     else FlaxT5EncoderModel)
        # Prefer native Flax weights, then convert local PyTorch weights.
        for from_pt in (False, True):
            try:
                model = model_cls.from_pretrained(
                    self.model_path, from_pt=from_pt, local_files_only=True)
                return model, model.params
            except Exception as exc:  # pragma: no cover - checkpoint specific
                errors.append(f"from_pt={from_pt}: {exc}")
        if not self.joint_planner:
            # A full seq2seq checkpoint exposes encoder/shared weights.  Build
            # an encoder-only module with the same config and reuse them.
            try:
                full = FlaxT5ForConditionalGeneration.from_pretrained(
                    self.model_path, from_pt=True, local_files_only=True)
                model = FlaxT5EncoderModel(full.config)
                params = {"shared": full.params["shared"],
                          "encoder": full.params["encoder"]}
                return model, params
            except Exception as exc:  # pragma: no cover
                errors.append(f"full-seq2seq fallback: {exc}")
        raise RuntimeError(
            "Could not load a Flax T5 model from the local checkpoint. "
            "Provide a local Flan-T5-small directory with tokenizer and "
            "either Flax or PyTorch weights. Details: " + " | ".join(errors))

    def init_params(self, rng):
        return self.params

    @staticmethod
    def _encoder_params(params):
        if "encoder" in params and "shared" in params:
            return {"shared": params["shared"], "encoder": params["encoder"]}
        return params

    def encode(self, params, ids, mask, train: bool = False,
               dropout_rng=None):
        params = self._encoder_params(params)
        if self.joint_planner:
            out = self.model.encode(
                input_ids=ids, attention_mask=mask, train=train, params=params,
                dropout_rng=dropout_rng)
        else:
            out = self.model(
                input_ids=ids, attention_mask=mask, train=train, params=params,
                dropout_rng=dropout_rng)
        return getattr(out, "last_hidden_state", out[0])

    def generate_plan_text(self, params, instruction: str,
                           max_new_tokens: int = 128) -> str:
        if not self.joint_planner:
            raise ValueError("plan generation requires the seq2seq model")
        import jax.numpy as jnp

        encoded = self.tokenizer(
            PLAN_PROMPT + str(instruction), return_tensors="np",
            truncation=True, max_length=self.planner_max_length)
        out = self.model.generate(
            input_ids=jnp.asarray(encoded["input_ids"]),
            attention_mask=jnp.asarray(encoded["attention_mask"]),
            params=params, max_new_tokens=int(max_new_tokens),
            do_sample=False, num_beams=1)
        sequences = getattr(out, "sequences", out)
        return self.tokenizer.decode(
            jnp.asarray(sequences)[0], skip_special_tokens=True)

    def planning_loss(self, params, batch, train: bool = False,
                      dropout_rng=None):
        if not self.joint_planner:
            raise ValueError("joint planning loss requires the seq2seq model")
        import jax.numpy as jnp

        labels = jnp.asarray(batch["labels"])
        _, decoder_input_ids = teacher_forced_decoder_inputs(
            labels, self.pad_token_id,
            self.model.config.decoder_start_token_id)
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        out = self.model(
            input_ids=jnp.asarray(batch["input_ids"]),
            attention_mask=jnp.asarray(batch["attention_mask"]),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            train=train, params=params, dropout_rng=dropout_rng)
        return t5_teacher_forced_loss(out.logits, labels)


def build_text_encoder(config: TextConfig):
    if config.encoder == "synthetic":
        return SyntheticTextEncoder(output_dim=32)
    if config.encoder == "none":
        return UnavailableTextEncoder(
            "TextEncoder='none' but the active VITA objective requires "
            "clause/object/reference conditioning")
    if config.encoder == "t5_flax":
        try:
            return TransformersFlaxT5Encoder(
                config.model_path, max_length=config.max_length,
                joint_planner=config.joint_planner,
                planner_max_length=config.planner_max_length,
                planner_labels_max_length=config.planner_labels_max_length)
        except Exception as exc:
            return UnavailableTextEncoder(
                f"Flax T5 adapter unavailable: {exc}")
    raise ValueError(f"unknown text encoder {config.encoder!r}")
