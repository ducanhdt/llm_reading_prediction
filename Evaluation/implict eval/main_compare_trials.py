import os
import argparse

# Prevent debugger freeze
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"

from models.compare_att import CompareAttention


# =========================================================
# Paths
# =========================================================

DATA_ROOT = "/kaggle/working/oasstetc_data/"
GPT2_ROOT = "/kaggle/working/gpt2_models"
LORA_ROOT = "/kaggle/input/fix-duration-model/trained_models_fix_duration"


# =========================================================
# Model discovery
# =========================================================

def discover_gpt2_models():
    models = {
        "gpt2": "causalLM",
        "gpt2-medium": "causalLM",
    }

    for size in ["small", "medium"]:
        size_root = os.path.join(GPT2_ROOT, size)
        if not os.path.isdir(size_root):
            continue

        for d in sorted(os.listdir(size_root)):
            model_dir = os.path.join(size_root, d)
            if not os.path.isdir(model_dir):
                continue

            step_dirs = [s for s in os.listdir(model_dir) if s.startswith("model_step_")]
            if not step_dirs:
                continue

            step = sorted(step_dirs)[-1]
            short_name = f"{d}_{step}"
            models[short_name] = "causalLM"

    return models


def discover_lora_models():
    models = {}

    if not os.path.isdir(LORA_ROOT):
        return models

    for d in sorted(os.listdir(LORA_ROOT)):
        full = os.path.join(LORA_ROOT, d)
        if os.path.isdir(full):
            models[d] = "causalLM"

    return models


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_completed", default=False)
    parser.add_argument("--folder_attention", type=str, default="attention")
    parser.add_argument("--only_gpt2", action="store_true")
    parser.add_argument("--only_lora", action="store_true")
    args = parser.parse_args()

    filter_completed = str(args.filter_completed).lower() == "true"
    folder_attention = args.folder_attention

    print(f"[INFO] filter_completed = {filter_completed}")
    print(f"[INFO] folder_attention = {folder_attention}")

    # --------------------------------------------------
    # Model registry
    # --------------------------------------------------
    models = {}

    if not args.only_lora:
        models.update(discover_gpt2_models())

    if not args.only_gpt2:
        models.update(discover_lora_models())

    print(f"[INFO] Models detected ({len(models)}):")
    for m in models:
        print("  -", m)

    # --------------------------------------------------
    # Gaze features
    # --------------------------------------------------
    gaze_features = [
        "first_fix_duration_n",
    ]

    # ==================================================
    # Compare per trial (all)
    # ==================================================
    for gaze_feature in gaze_features:
        for model_name, model_type in models.items():
            print(f"[RUN] {model_name} | {gaze_feature}")
            CompareAttention(
                model_name=model_name,
                model_type=model_type,
                path=DATA_ROOT,
            ).compute_sc_model_per_trial(
                gaze_feature=gaze_feature,
                filter_completed=filter_completed,
                folder_attention=folder_attention,
            )

        CompareAttention.compare_between_models_per_trials(
            folder=os.path.join(DATA_ROOT, folder_attention),
            gaze_feature=gaze_feature,
            filter_completed=filter_completed,
        )

    # ==================================================
    # Compare chosen vs rejected
    # ==================================================
    for gaze_feature in gaze_features:
        for model_name, model_type in models.items():
            print(f"[CR] {model_name} | {gaze_feature}")
            CompareAttention(
                model_name=model_name,
                model_type=model_type,
                path=DATA_ROOT,
            ).compute_sc_model_chosenrejected_per_trials(
                gaze_feature=gaze_feature,
                filter_completed=filter_completed,
                folder_attention=folder_attention,
            )

        CompareAttention.compare_between_models_chosenrejected_per_trials(
            folder=os.path.join(DATA_ROOT, folder_attention),
            gaze_feature=gaze_feature,
            filter_completed=filter_completed,
        )


if __name__ == "__main__":
    main()