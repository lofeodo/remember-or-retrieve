"""Programmatic quality gates for generated rows. Each gate returns a reason string when a row
fails, or None when it passes, so the report can count drops per gate."""

import re

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_BANNED = re.compile(
    r"\b(the (excerpt|passage|text|table|chunk)|according to|in the excerpt|above)\b", re.IGNORECASE
)


def alnum(s: str) -> str:
    """Lowercase, alphanumerics only. Makes quote matching robust to table spacing and quotes."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def norm_words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def numbers(s: str) -> set[str]:
    out = set()
    for m in _NUM.findall(s):
        m = m.replace(",", "").rstrip(".")
        out.add(m.rstrip("0").rstrip(".") if "." in m else m)
    return out


def evidence_in(evidence: str, chunk_text: str, min_chars: int = 12) -> bool:
    e = alnum(evidence)
    return len(e) >= min_chars and e in alnum(chunk_text)


def common_gates(question: str, answer: str, facts: list[str]) -> str | None:
    if not (10 <= len(question) <= 400):
        return "question_length"
    if not answer.strip() or len(answer) > 800:
        return "answer_length"
    if _BANNED.search(question) or _BANNED.search(answer):
        return "refers_to_excerpt"
    if not facts or any(not f.strip() for f in facts):
        return "no_reference_facts"
    if not any(alnum(f) in alnum(answer) for f in facts if alnum(f)):
        return "facts_not_in_answer"
    return None


def single_hop_gate(pair: dict, chunk_text: str) -> str | None:
    if (r := common_gates(pair["question"], pair["answer"], pair["reference_facts"])) is not None:
        return r
    if not evidence_in(pair["evidence"], chunk_text):
        return "evidence_not_verbatim"
    if numbers(pair["answer"]) - numbers(pair["evidence"]) - numbers(pair["question"]):
        return "number_not_in_evidence"
    return None


def multi_hop_gate(row: dict, text_a: str, text_b: str) -> str | None:
    if not row["feasible"]:
        return "not_feasible"
    if (r := common_gates(row["question"], row["answer"], row["reference_facts"])) is not None:
        return r
    if not evidence_in(row["evidence_a"], text_a) or not evidence_in(row["evidence_b"], text_b):
        return "evidence_not_verbatim"
    # The result of the arithmetic is allowed to be new; any more unexplained numbers than a
    # result and one intermediate value means an operand was invented.
    unexplained = (
        numbers(row["answer"])
        - numbers(row["evidence_a"])
        - numbers(row["evidence_b"])
        - numbers(row["question"])
    )
    if len(unexplained) > 2:
        return "operands_not_in_evidence"
    return None


def solver_agrees(reference_facts: list[str], solver_answer: str) -> bool:
    """Every number in the reference facts appears in the independent solver's answer, and
    every non-numeric fact appears as text."""
    sol_nums, sol_alnum = numbers(solver_answer), alnum(solver_answer)
    for fact in reference_facts:
        nums = numbers(fact)
        if nums:
            if not nums <= sol_nums:
                return False
        elif alnum(fact) and alnum(fact) not in sol_alnum:
            return False
    return True


def ngrams(s: str, n: int = 5) -> set[tuple[str, ...]]:
    w = norm_words(s)
    return {tuple(w[i : i + n]) for i in range(max(len(w) - n + 1, 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0
