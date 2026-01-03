import random
import torch

MAX_LEN = 1024


def judge(tok, model, question, A, B):
    """
    Pairwise preference judging: A vs B
    Returns: "A" or "B"
    """

    # random flip to avoid position bias
    if random.random() < 0.5:
        pA, pB = A, B
        flip = False
    else:
        pA, pB = B, A
        flip = True

    text = f"Which answer is better?\n\nA: {pA}\n\nB: {pB}\n\nAnswer:"
    ids = tok.encode(text, add_special_tokens=False)[:MAX_LEN]
    ids_t = torch.tensor([ids], device=model.device)

    with torch.no_grad():
        embeds = model.get_input_embeddings()(ids_t)
        logits = model(inputs_embeds=embeds).logits[0, -1]

    token_A = tok.encode("A", add_special_tokens=False)[-1]
    token_B = tok.encode("B", add_special_tokens=False)[-1]

    pred = "A" if logits[token_A] > logits[token_B] else "B"
    if flip:
        pred = "B" if pred == "A" else "A"

    return pred


def evaluate_model(tok, model, triples):
    """
    triples: list[(question, good, bad)]
    """
    correct = 0
    for q, good, bad in triples:
        pred = judge(tok, model, q, bad, good)
        if pred == "B":  # good is B
            correct += 1
    return correct / len(triples)