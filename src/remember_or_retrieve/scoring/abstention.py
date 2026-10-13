"""Decide whether an answer is an abstention, and label the answerable/abstained cross.

The RAG pipeline reports its own `answerable` flag, which wins. Closed-book systems (the
fine-tune, Mistral, Cohere) have no flag, so an abstention is the trained phrase.
"""

import re

from remember_or_retrieve.datagen.prompts import ABSTAIN_PHRASE
from remember_or_retrieve.scoring.schema import (
    CORRECT_ABSTENTION,
    INCORRECTLY_ABSTAINED,
    INCORRECTLY_ANSWERED,
)

_WORDS = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORDS.findall(text.lower()))


_PHRASE = _norm(ABSTAIN_PHRASE)


def is_abstention(answer: str) -> bool:
    """True if the answer is the abstention phrase, optionally followed by a short remark."""
    a = _norm(answer)
    return bool(a) and a.startswith(_PHRASE)


def abstention_label(expected_answerable: bool, abstained: bool) -> str | None:
    """The label decided without the judge, or None when both say answerable (judge decides)."""
    if expected_answerable and abstained:
        return INCORRECTLY_ABSTAINED
    if not expected_answerable:
        return CORRECT_ABSTENTION if abstained else INCORRECTLY_ANSWERED
    return None
