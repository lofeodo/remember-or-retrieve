"""Gemma 4 tokenizer adapter for the shared SFT encoding (`training.data.Tokenizer` protocol).

The chat format is Gemma 4's own: a native system turn, then `<|turn>user ... <turn|>`, then the
generation prompt `<|turn>model\n`. The answer target ends with the end-of-turn token `<turn|>` so
the model learns to stop, exactly as the Qwen3 stage learns `<|im_end|>`.

The heavy `gemma` import is lazy and the inner tokenizer is injectable, so the logic is unit-tested
without JAX.
"""

from typing import Protocol

BOS_ID = 2
PAD_ID = 0
END_OF_TURN = "<turn|>"
END_OF_TURN_ID = 106


class Inner(Protocol):
    """The slice of `gm.text.Gemma4Tokenizer` used here."""

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


def render_prompt(messages: list[dict]) -> str:
    """The Gemma 4 chat text for `messages` plus the model generation prompt."""
    parts = []
    for m in messages:
        role = {"system": "system", "user": "user", "assistant": "model"}[m["role"]]
        parts.append(f"<|turn>{role}\n{m['content']}{END_OF_TURN}\n")
    return "".join(parts) + "<|turn>model\n"


class GemmaTokenizer:
    """Implements `training.data.Tokenizer` for Gemma 4 (plus `decode` for generation)."""

    pad_token_id = PAD_ID
    end_token = END_OF_TURN
    end_token_id = END_OF_TURN_ID

    def __init__(self, inner: Inner):
        self.inner = inner

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        return self.inner.encode(render_prompt(messages), add_bos=True)

    def encode(self, text: str) -> list[int]:
        """Token ids for answer text. A trailing `<turn|>` becomes the single end-of-turn id
        (explicitly, so answer text can never be parsed as a control token elsewhere)."""
        if text.endswith(END_OF_TURN):
            return self.inner.encode(text.removesuffix(END_OF_TURN)) + [END_OF_TURN_ID]
        return self.inner.encode(text)

    def decode(self, ids: list[int]) -> str:
        cut = ids.index(END_OF_TURN_ID) if END_OF_TURN_ID in ids else len(ids)
        return self.inner.decode(ids[:cut]).strip()


def load_tokenizer(cfg: dict) -> GemmaTokenizer:
    """The real tokenizer, downloaded anonymously from the public Gemma bucket (needs no token)."""
    from gemma import gm

    return GemmaTokenizer(gm.text.Gemma4Tokenizer(path=cfg["jax"]["model"]["tokenizer"]))
