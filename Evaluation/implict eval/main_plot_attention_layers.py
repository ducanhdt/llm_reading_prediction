import os
import matplotlib

# --------------------------------------------------
# Headless backend
# --------------------------------------------------
matplotlib.use("Agg")

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"

from models.compare_att import CompareAttention


# ==================================================
# Config
# ==================================================

PATH = "/kaggle/working/data/"
ATTENTION_ROOT = os.path.join(PATH, "attention")

GAZE_FEATURES = [
    "first_fix_duration_n",
]

FILTER_COMPLETED = True
ATTENTION_FOLDER = "attention"


# ==================================================
# Model label helper (for logging only)
# ==================================================

def short_name(name: str):
    if "gemma3-1b" in name:
        base = "Gemma-1B"
    elif "qwen0.6b" in name:
        base = "Qwen-0.6B"
    else:
        base = name

    if "KL-5-" in name:
        kl = "KL=5"
    elif "KL-50-" in name:
        kl = "KL=50"
    elif "KL-500-" in name:
        kl = "KL=500"
    else:
        kl = "KL=?"

    if "lr-0.0001" in name:
        lr = "lr=1e-4"
    elif "lr-1e-05" in name:
        lr = "lr=1e-5"
    else:
        lr = "lr=?"

    return f"{base}\n{kl}\n{lr}"


# ==================================================
# Auto-detect models from attention folder
# ==================================================

models = {}

for d in sorted(os.listdir(ATTENTION_ROOT)):
    if d == "results":
        continue

    full = os.path.join(ATTENTION_ROOT, d)
    if os.path.isdir(full):
        models[d] = "causalLM"

print("\n[INFO] Models detected:")
for m in models:
    print(" ", short_name(m))


# ==================================================
# Plot layer-wise attention correlations
# ==================================================

for model_name, model_type in models.items():
    print(f"\n[Plotting layers] {short_name(model_name)}")

    comp = CompareAttention(
        model_name=model_name,
        model_type=model_type,
        path=PATH,
    )

    # -------- all trials --------
    comp.plot_attention_all_trials(
        gaze_features=GAZE_FEATURES,
        attention_folder=ATTENTION_FOLDER,
        filter_completed=FILTER_COMPLETED,
    )

    # -------- chosen vs rejected --------
    comp.plot_attention_chosenrejected(
        gaze_features=GAZE_FEATURES,
        attention_folder=ATTENTION_FOLDER,
        filter_completed=FILTER_COMPLETED,
    )

print("\n[Done] All attention layer plots saved.")