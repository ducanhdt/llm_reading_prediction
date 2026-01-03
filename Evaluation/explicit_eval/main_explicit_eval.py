from build_triples import load_oasst_triples
from model_registry import load_single_model
from explicit_eval import evaluate_model
from plot_accuracy import plot_accuracy

from model_registry import MODEL_REGISTRY


def run(models, title):
    triples = load_oasst_triples()

    results = {}
    for name in models:
        print(f"\nEvaluating {name}")
        tok, model = load_single_model(MODEL_REGISTRY[name])
        acc = evaluate_model(tok, model, triples)
        results[name] = acc
        print(name, "→", acc)

    plot_accuracy(results, title)


if __name__ == "__main__":
    run(
        models=["baseline_small", "small_300", "small_800"],
        title="Small GPT-2 Models",
    )