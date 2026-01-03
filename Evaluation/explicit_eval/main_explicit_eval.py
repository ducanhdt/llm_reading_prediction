from explicit_eval.build_triples import build_oasst_triples
from explicit_eval.model_registry import load_model
from explicit_eval.explicit_eval import evaluate_model
from explicit_eval.plot_accuracy import plot_accuracy

MODELS = {
    "gpt2": {"type": "hf", "path": "gpt2"},
    "gpt2-medium": {"type": "hf", "path": "gpt2-medium"},
    "small_ft": {"type": "full", "path": "/kaggle/working/small_step800"},
    "gemma_peft": {
        "type": "peft",
        "base": "google/gemma-3-1b-it",
        "path": "/kaggle/working/gemma3-1b-reading-time-KL-5-lr-1e-05",
    },
}

triples = build_oasst_triples(max_samples=500)

results = {}
for name, cfg in MODELS.items():
    print(f"\n[Eval] {name}")
    tok, model = load_model(cfg)
    acc = evaluate_model(tok, model, triples)
    results[name] = acc
    print(f"{name}: {acc:.3f}")

plot_accuracy(
    results,
    title="Explicit Preference Evaluation",
    save_path="figures/explicit_eval_accuracy.png",
)
