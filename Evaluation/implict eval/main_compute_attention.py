import os
import argparse

# Prevent debugger freeze in Kaggle
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"

from models.human_att import HumanAttentionExtractor
from models.model_att import ModelAttentionExtractor


# =========================================================
# Config
# =========================================================

DATA_ROOT = "oasstetc_data/"
GPT2_LOCAL_ROOT = "/kaggle/working/gpt2_models"
LORA_ROOT = "/kaggle/input/fix-duration-model/trained_models_fix_duration"

DEFAULT_USER_SETS = [8]


# =========================================================
# Model discovery
# =========================================================

def discover_gpt2_models():
    """
    Discover:
      - GPT-2 baseline
      - GPT-2 local finetuned checkpoints
    """
    models = {
        "gpt2": "causalLM",
        "gpt2-medium": "causalLM",
    }

    model_dirs = [
        "medium/all_medium_fd_0_kl_config_42",
        "small/all_small_fd_config_8",
        "medium/all_medium_ffd_0_kl_config_42",
        "small/all_small_ffd_config_8",
    ]

    for d in model_dirs:
        model_dir = os.path.join(GPT2_LOCAL_ROOT, d)
        if not os.path.isdir(model_dir):
            continue

        step_subdirs = sorted(
            f for f in os.listdir(model_dir) if f.startswith("model_step_")
        )
        if not step_subdirs:
            print(f"[WARN] no checkpoint in {model_dir}")
            continue

        step = step_subdirs[-1]
        ckpt_path = os.path.join(model_dir, step)
        models[ckpt_path] = "causalLM"

    return models


def discover_lora_models():
    """
    Discover Gemma / Qwen LoRA adapters
    """
    models = {}

    if not os.path.isdir(LORA_ROOT):
        print(f"[WARN] LoRA root not found: {LORA_ROOT}")
        return models

    for name in os.listdir(LORA_ROOT):
        full_path = os.path.join(LORA_ROOT, name)
        if os.path.isdir(full_path):
            models[full_path] = "causalLM"

    return models


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", default=False, help="use reward attention")
    parser.add_argument("--only_lora", action="store_true", help="run only LoRA models")
    parser.add_argument("--only_gpt2", action="store_true", help="run only GPT-2 models")
    args = parser.parse_args()

    reward = str(args.reward).lower() == "true"
    print(f"[INFO] reward = {reward}")

    # --------------------------------------------------
    # Model registry
    # --------------------------------------------------
    models = {}

    if not args.only_lora:
        models.update(discover_gpt2_models())

    if not args.only_gpt2:
        models.update(discover_lora_models())

    print(f"[INFO] total models detected: {len(models)}")

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    human_extractor = HumanAttentionExtractor()

    for model_name, model_type in models.items():
        print(f"\n==============================")
        print(f"[MODEL] {model_name}")
        print(f"==============================")

        att_extractor = ModelAttentionExtractor(model_name, model_type)

        for user_set in DEFAULT_USER_SETS:
            print(f"[SET] user set {user_set}")

            # ---------- naming ----------
            if model_name in ["gpt2", "gpt2-medium"]:
                safe_model_name = model_name
            else:
                safe_model_name = os.path.basename(model_name.rstrip("/"))

            if reward:
                out_dir = os.path.join(
                    DATA_ROOT, "attention_reward", safe_model_name, f"set_{user_set}"
                )
            else:
                out_dir = os.path.join(
                    DATA_ROOT, "attention", safe_model_name, f"set_{user_set}"
                )

            os.makedirs(out_dir, exist_ok=True)

            # ---------- load texts ----------
            text_dir = os.path.join(
                DATA_ROOT, "gaze_features_real", f"set_{user_set}"
            )
            texts_trials = human_extractor.load_texts(text_dir)
            prompts_df = HumanAttentionExtractor.load_trial_prompts()

            # ---------- compute ----------
            if reward:
                attention_trials = att_extractor.extract_attention_reward(
                    texts_trials, texts_promps=prompts_df
                )
            else:
                attention_trials = att_extractor.extract_attention(
                    texts_trials, word_level=True
                )

            # ---------- save ----------
            att_extractor.save_attention_df(
                attention_trials,
                texts_trials=texts_trials,
                path_folder=out_dir,
            )

            print(f"[SAVED] → {out_dir}")


if __name__ == "__main__":
    main()