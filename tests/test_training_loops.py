"""Fast tests for the loop internals that need torch (skipped in the dev-only CI job)."""

import pytest

torch = pytest.importorskip("torch")

from remember_or_retrieve.training import dpo, sft  # noqa: E402


def test_lr_warms_up_then_decays_to_ten_percent():
    lrs = [sft.lr_at(s, 100, 10, 1.0) for s in range(100)]
    assert lrs[0] < lrs[9] <= 1.0 and lrs[10] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.1, abs=0.01)
    assert all(a >= b - 1e-9 for a, b in zip(lrs[10:], lrs[11:], strict=False))


def test_dpo_loss_is_ln2_at_the_reference_and_falls_when_chosen_gains():
    z = torch.zeros(3)
    loss, margin = dpo.dpo_loss(z, z, z, z, beta=0.1)
    assert torch.allclose(loss, torch.full((3,), 0.6931), atol=1e-3) and margin.abs().sum() == 0
    better, m = dpo.dpo_loss(z + 5, z - 5, z, z, beta=0.1)
    assert (better < loss).all() and (m > 0).all()
    worse, _ = dpo.dpo_loss(z - 5, z + 5, z, z, beta=0.1)
    assert (worse > loss).all()


def test_sequence_logps_sums_only_answer_tokens():
    class Net:
        def __call__(self, input_ids, attention_mask):
            logits = torch.zeros(*input_ids.shape, 4)
            logits[..., 2] = 10.0  # always predicts token 2
            return type("O", (), {"logits": logits})()

    import contextlib

    batch = {
        "input_ids": torch.tensor([[1, 1, 2, 2]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 2, 2]]),
    }
    lp = dpo.sequence_logps(Net(), batch, contextlib.nullcontext())
    assert lp.shape == (1,) and abs(lp.item()) < 0.02  # near-certain prediction: log-prob ~ 0
