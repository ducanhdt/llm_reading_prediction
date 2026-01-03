from datasets import load_dataset
from collections import defaultdict
import random

MAX_LEN = 1024
MAX_ANS_CHAR = 3000


def build_prompt(question, A, B):
    return f"""You are a fair evaluator.

Question:
{question}

Answer A:
{A}

Answer B:
{B}

Which answer is better? Reply with only 'A' or 'B'.
"""


def load_oasst_triples(seed=42, max_triples=500):
    """
    Returns: list[(prompt, good, bad)]
    """
    ds = load_dataset("OpenAssistant/oasst1")["train"]
    ds = ds.filter(lambda x: x["lang"] == "en" and x["deleted"] is False)

    prompters = [m for m in ds if m["role"] == "prompter"]
    assistants = [m for m in ds if m["role"] == "assistant"]

    prompt_text = {m["message_id"]: m["text"] for m in prompters}

    replies = defaultdict(list)
    for m in assistants:
        pid = m["parent_id"]
        if pid in prompt_text and m["rank"] is not None:
            replies[pid].append(m)

    all_triples = []
    for pid, reps in replies.items():
        reps = sorted(reps, key=lambda r: r["rank"])
        if len(reps) < 2:
            continue
        all_triples.append(
            (prompt_text[pid], reps[0]["text"], reps[-1]["text"])
        )

    random.seed(seed)
    random.shuffle(all_triples)
    return all_triples[:max_triples]