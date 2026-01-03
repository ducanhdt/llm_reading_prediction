from datasets import load_dataset
from collections import defaultdict
import random


def build_oasst_triples(max_samples=500, seed=42):
    ds = load_dataset("OpenAssistant/oasst1")["train"]
    ds = ds.filter(lambda x: x["lang"] == "en" and not x["deleted"])

    prompters = [m for m in ds if m["role"] == "prompter"]
    assistants = [m for m in ds if m["role"] == "assistant"]

    prompt_text = {m["message_id"]: m["text"] for m in prompters}
    replies = defaultdict(list)

    for m in assistants:
        if m["parent_id"] in prompt_text and m["rank"] is not None:
            replies[m["parent_id"]].append(m)

    triples = []
    for pid, reps in replies.items():
        reps = sorted(reps, key=lambda r: r["rank"])
        if len(reps) >= 2:
            triples.append(
                (prompt_text[pid], reps[0]["text"], reps[-1]["text"])
            )

    random.seed(seed)
    random.shuffle(triples)
    return triples[:max_samples]