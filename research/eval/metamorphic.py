"""Small, reproducible choice-invariance diagnostics (issue #244).

Run from the checkout: python -m research.eval.metamorphic --help.
Pure transformations/metrics can also be used without a model or gold labels.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any



@dataclass
class MetamorphicCase:
    """One canonical choice decision in the existing harness input format.

    Descriptions must stand alone; state/instructions must not refer to labels
    or positions. Nonempty descriptions are necessary, not proof of equivalence.
    """

    state: Any
    questions: dict
    gold_index: int | None = None

    def __post_init__(self):
        if len(self.questions) != 1:
            raise ValueError("expected exactly one choice question per case")
        question = next(iter(self.questions.values()))
        criteria = question.get("criteria")
        if question.get("type") != "choice" or not isinstance(criteria, dict):
            raise ValueError("expected choice criteria as a dictionary")
        if len(criteria) < 2:
            raise ValueError("at least two options are required")
        if any(not isinstance(k, str) or not k.strip() for k in criteria):
            raise ValueError("option keys must be nonempty strings")
        if any(v is None or (isinstance(v, str) and not v.strip())
               for v in criteria.values()):
            raise ValueError("neutral labels require standalone nonempty descriptions")
        if self.gold_index is not None and (
                type(self.gold_index) is not int or not 0 <= self.gold_index < len(criteria)):
            raise ValueError("gold index outside canonical options")

    @property
    def option_keys(self):
        return list(next(iter(self.questions.values()))["criteria"])

    def as_pair(self):
        return self.state, self.questions


@dataclass
class MetamorphicVariant:
    """Transformed input plus an explicit bidirectional semantic label mapping."""

    kind: str
    case: MetamorphicCase
    canonical_to_transformed: dict[str, str]

    @property
    def transformed_to_canonical(self):
        return {v: k for k, v in self.canonical_to_transformed.items()}

    def as_record(self):
        mapping = self.canonical_to_transformed
        inverse = self.transformed_to_canonical
        keys = list(mapping)
        if len(inverse) != len(mapping) or set(inverse) != set(self.case.option_keys):
            raise ValueError("label mapping must be a bijection over transformed options")
        return {"kind": self.kind, "case": self.case.as_pair(),
                "canonical_to_transformed": dict(mapping),
                "transformed_to_canonical": inverse,
                "canonical_indices": [keys.index(inverse[k]) for k in self.case.option_keys]}


def _transform(case, kind, order, labels):
    questions = deepcopy(case.questions)
    question = next(iter(questions.values()))
    criteria = question["criteria"]
    keys = case.option_keys
    mapping = dict(zip(keys, labels))
    question["criteria"] = {mapping[keys[i]]: deepcopy(criteria[keys[i]]) for i in order}
    gold = None if case.gold_index is None else order.index(case.gold_index)
    return MetamorphicVariant(kind, MetamorphicCase(deepcopy(case.state), questions, gold), mapping)


def _baseline(case):
    return _transform(case, "baseline", list(range(len(case.option_keys))), case.option_keys)


def canonicalize(probabilities, canonical_indices):
    """Validate a probability vector and restore canonical semantic ordering."""
    values = [float(p) for p in probabilities]
    n = len(values)
    if not n or any(type(i) is not int for i in canonical_indices) or sorted(canonical_indices) != list(range(n)):
        raise ValueError("probabilities and canonical mapping must be a bijection")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("probabilities must sum to one")
    restored = [0.0] * n
    for p, index in zip(values, canonical_indices):
        restored[index] = p
    return restored


