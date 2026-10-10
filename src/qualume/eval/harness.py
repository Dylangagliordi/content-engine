import json
from dataclasses import dataclass


@dataclass
class EvalResult:
    stage: str
    golden_set_version: str
    prompt_version: str
    total: int
    passed: int

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


def load_golden_set(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_constraint_checks(generated_text: str, example: dict) -> bool:
    """Deterministic checks only at this stage — no LLM judge yet.
    Returns True if generated_text satisfies must_include/must_avoid."""
    text_lower = generated_text.lower()
    for must in example.get("must_include", []):
        if must.lower() not in text_lower:
            return False
    for avoid in example.get("must_avoid", []):
        if avoid.lower() in text_lower:
            return False
    return True
