"""Model and tokenizer construction: the real Qwen3-8B, a LoRA wrapper, and the smoke-test model.

Everything is loaded with `trust_remote_code=False` from a pinned revision, safetensors only.
"""

import torch

from remember_or_retrieve.training.data import HFTokenizer


def load_tokenizer(model_cfg: dict) -> HFTokenizer:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_cfg["name"], revision=model_cfg["revision"], trust_remote_code=False
    )
    return HFTokenizer(tok)


def torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[name]


def load_pretrained(model_cfg: dict, source: str | None = None):
    """Qwen3-8B, or a local checkpoint dir as `source` (DPO on top of SFT), on CPU in bf16.
    DDP / FSDP2 move it to the GPU when wrapping."""
    from transformers import AutoModelForCausalLM

    kwargs = {} if source else {"revision": model_cfg["revision"]}
    return AutoModelForCausalLM.from_pretrained(
        source or model_cfg["name"],
        dtype=torch_dtype(model_cfg["dtype"]),
        attn_implementation="sdpa",
        trust_remote_code=False,
        use_safetensors=True,
        **kwargs,
    )


def load_trained(model_cfg: dict, path: str, smoke_cfg: dict | None = None, seed: int = 0):
    """A trained checkpoint as one plain model: a full-SFT model dir as is, or a LoRA adapter dir
    merged into its base. With `smoke_cfg`, the base is the seeded tiny smoke model."""
    from pathlib import Path

    if not (Path(path) / "adapter_config.json").exists():
        return load_pretrained(model_cfg, path)
    from peft import PeftModel

    if smoke_cfg is not None:
        from transformers import AutoTokenizer

        base = tiny_model(smoke_cfg, len(AutoTokenizer.from_pretrained(path)), seed)
    else:
        base = load_pretrained(model_cfg)
    return PeftModel.from_pretrained(base, path).merge_and_unload()


def tiny_model(smoke_cfg: dict, vocab_size: int, seed: int):
    """A small randomly initialised Qwen3-architecture model in fp32 for the CPU smoke test.

    Seeded, so every rank (and a 1-rank reference run) starts from identical weights.
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=smoke_cfg["hidden_size"],
        intermediate_size=smoke_cfg["intermediate_size"],
        num_hidden_layers=smoke_cfg["num_hidden_layers"],
        num_attention_heads=smoke_cfg["num_attention_heads"],
        num_key_value_heads=smoke_cfg["num_key_value_heads"],
        head_dim=smoke_cfg["head_dim"],
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        use_cache=False,
    )
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(cfg).to(torch.float32)


def build_lora(model, lora_cfg: dict):
    """Wrap `model` with LoRA adapters on all linear projections."""
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=list(lora_cfg["target_modules"]),
            task_type="CAUSAL_LM",
        ),
    )


def count_params(model) -> tuple[int, int]:
    """(total, trainable) parameter counts (global shapes, so FSDP2 shards are not undercounted)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
