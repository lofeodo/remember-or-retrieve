"""Thin Anthropic wrapper for build-time generation: Batch API, resume, cost accounting.

The API key comes from the ANTHROPIC_API_KEY environment variable only. Nothing here prints it.
Sonnet 5.5 rejects forced tool_choice, so structured JSON output (output_config.format) is used.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Request:
    custom_id: str  # Batch API ids allow only [A-Za-z0-9_-], max 64 chars
    model: str
    system: str
    user: str
    schema: dict
    effort: str = "low"
    thinking: str = "off"  # "off" or "adaptive"
    max_tokens: int = 2048
    est_out: int = 400  # expected output tokens, used only for the pre-submit estimate
    meta: dict = field(default_factory=dict)  # carried through to results, never sent


def build_params(req: Request) -> dict:
    params = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "system": req.system,
        "messages": [{"role": "user", "content": req.user}],
        "output_config": {
            "effort": req.effort,
            "format": {"type": "json_schema", "schema": req.schema},
        },
    }
    if req.thinking == "off":
        # Thinking cannot be disabled on Sonnet 5.5 (400); between_tools is its "off".
        params["thinking"] = (
            {"type": "between_tools"} if "sonnet" in req.model else {"type": "disabled"}
        )
    return params


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.6) + 1


def estimate_cost(requests: list[Request], prices: dict, batch_discount: float) -> dict:
    """Heuristic pre-submit estimate (chars/3.6 for input, declared est_out for output)."""
    per_model: dict[str, dict] = {}
    for r in requests:
        m = per_model.setdefault(r.model, {"requests": 0, "input_tokens": 0, "output_tokens": 0})
        m["requests"] += 1
        m["input_tokens"] += estimate_tokens(r.system + r.user + json.dumps(r.schema))
        m["output_tokens"] += r.est_out
    total = 0.0
    for model, m in per_model.items():
        p = prices[model]
        m["usd"] = round(
            (m["input_tokens"] * p["input"] + m["output_tokens"] * p["output"])
            / 1e6
            * batch_discount,
            4,
        )
        total += m["usd"]
    return {"per_model": per_model, "usd": round(total, 4)}


def usage_cost(model: str, input_tokens: int, output_tokens: int, prices: dict, discount: float):
    p = prices[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1e6 * discount


def make_client():
    import anthropic

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set (export it in your shell; never commit it)"
        )
    return anthropic.Anthropic()


def _extract(message) -> tuple[str, dict | None]:
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        return text, json.loads(text)
    except json.JSONDecodeError:
        return text, None


class Stage:
    """One stage of requests with on-disk state under raw_dir/<name>/ so runs resume."""

    def __init__(self, raw_dir: Path, name: str):
        self.dir = Path(raw_dir) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.requests_path = self.dir / "requests.jsonl"
        self.batch_path = self.dir / "batch_id.txt"
        self.results_path = self.dir / "results.jsonl"

    def save_requests(self, requests: list[Request]) -> None:
        with open(self.requests_path, "w", encoding="utf-8", newline="\n") as f:
            for r in requests:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def load_requests(self) -> list[Request]:
        with open(self.requests_path, encoding="utf-8") as f:
            return [Request(**json.loads(line)) for line in f if line.strip()]

    def load_results(self) -> dict[str, dict]:
        if not self.results_path.exists():
            return {}
        with open(self.results_path, encoding="utf-8") as f:
            return {(d := json.loads(line))["custom_id"]: d for line in f if line.strip()}


def submit(client, stage: Stage, requests: list[Request], max_usd: float, prices, discount) -> str:
    """Submit a stage as one batch. Refuses if already submitted or over budget."""
    if stage.batch_path.exists():
        return stage.batch_path.read_text().strip()
    est = estimate_cost(requests, prices, discount)
    if est["usd"] > max_usd:
        raise BudgetExceeded(f"estimate ${est['usd']:.2f} exceeds max_usd ${max_usd:.2f}")
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request as BatchRequest

    stage.save_requests(requests)
    batch = client.messages.batches.create(
        requests=[
            BatchRequest(
                custom_id=r.custom_id, params=MessageCreateParamsNonStreaming(**build_params(r))
            )
            for r in requests
        ]
    )
    stage.batch_path.write_text(batch.id)
    return batch.id


def collect(client, stage: Stage, prices, discount, wait: bool = True, poll_s: int = 60) -> dict:
    """Wait for the stage's batch, then write results.jsonl. Returns a status/cost summary."""
    batch_id = stage.batch_path.read_text().strip()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended" or not wait:
            break
        time.sleep(poll_s)
    if batch.processing_status != "ended":
        return {"status": batch.processing_status}
    reqs = {r.custom_id: r for r in stage.load_requests()}
    usd = 0.0
    counts = {"succeeded": 0, "failed": 0}
    with open(stage.results_path, "w", encoding="utf-8", newline="\n") as f:
        for item in client.messages.batches.results(batch_id):
            req = reqs[item.custom_id]
            row = {"custom_id": item.custom_id, "meta": req.meta}
            if item.result.type == "succeeded":
                msg = item.result.message
                text, data = _extract(msg)
                row.update(
                    data=data,
                    text=text if data is None else "",
                    stop_reason=msg.stop_reason,
                    model=msg.model,
                    input_tokens=msg.usage.input_tokens,
                    output_tokens=msg.usage.output_tokens,
                )
                usd += usage_cost(
                    req.model, row["input_tokens"], row["output_tokens"], prices, discount
                )
                counts["succeeded" if data is not None else "failed"] += 1
            else:
                row.update(data=None, error=item.result.type)
                counts["failed"] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"status": "ended", "usd": round(usd, 4), **counts}


def sync_call(client, req: Request) -> dict:
    """One synchronous call (used by the pilot). Returns data plus token usage."""
    msg = client.messages.create(**build_params(req))
    text, data = _extract(msg)
    return {
        "data": data,
        "text": text,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "stop_reason": msg.stop_reason,
    }
