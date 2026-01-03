import os
import re
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# Label helpers
# =========================================================

def format_gpt_label(model_id: str):
    # -------- model size --------
    if "medium" in model_id:
        model_line = "GPT-2 Medium"
    elif "small" in model_id:
        model_line = "GPT-2 Small"
    else:
        model_line = "GPT-2"

    # -------- config --------
    cfg = []
    if "ffd" in model_id:
        cfg.append("FFD")
    elif "_fd_" in model_id:
        cfg.append("FD")

    if "_kl_" in model_id:
        kl = model_id.split("_kl_")[1].split("_")[0]
        cfg.append(f"KL={kl}")

    cfg_line = ", ".join(cfg) if cfg else ""

    # -------- step --------
    step = ""
    if "model_step_" in model_id:
        step = model_id.split("model_step_")[1]

    step_line = f"step {step}" if step else ""

    return f"{model_line}\n{cfg_line}\n{step_line}".strip()


def format_lora_label(name: str):
    if "gemma" in name.lower():
        base = "Gemma-1B"
    elif "qwen" in name.lower():
        base = "Qwen-0.6B"
    else:
        return name

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


def smart_model_label(name: str):
    if "gpt2" in name.lower():
        return format_gpt_label(name)
    return format_lora_label(name)


# =========================================================
# Plotter
# =========================================================

class Plotter:

    # --------------------------------------------------
    # Generic bar plot (mean ± std)
    # --------------------------------------------------
    @staticmethod
    def plot_gaze_signal(path, df, gaze_signal, tag="", plot_std=True):
        categories = df.columns.tolist()
        means = [df.loc["mean", m] for m in categories]
        stds = [df.loc["std", m] for m in categories]

        data = sorted(zip(categories, means, stds), key=lambda x: x[1])
        categories, means, stds = zip(*data)

        colors = sns.color_palette("Blues", len(categories))
        x = np.arange(len(categories))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, means, color=colors, edgecolor="black")

        if plot_std:
            for i in range(len(x)):
                ax.fill_between(
                    [x[i] - 0.2, x[i] + 0.2],
                    means[i] - stds[i],
                    means[i] + stds[i],
                    color="gray",
                    alpha=0.3,
                )

        ax.set_ylabel("Correlation")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [smart_model_label(c) for c in categories],
            fontsize=11,
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)

        out_dir = os.path.join(path, "plots")
        os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{tag}_{gaze_signal}_barplot.png", dpi=300)
        plt.show()

    # --------------------------------------------------
    # Chosen vs rejected
    # --------------------------------------------------
    @staticmethod
    def plot_gaze_signal_chosenrejected(
        path, df, gaze_signal, tag="", plot_std=True, p_values={}
    ):
        feature_name = gaze_signal.split("_")[0]
        categories = df["chosen"].columns.tolist()

        means_c = [df["chosen"].loc["mean", m] for m in categories]
        stds_c = [df["chosen"].loc["std", m] for m in categories]
        means_r = [df["rejected"].loc["mean", m] for m in categories]
        stds_r = [df["rejected"].loc["std", m] for m in categories]

        data = sorted(zip(categories, means_c, stds_c, means_r, stds_r), key=lambda x: x[1])
        categories, means_c, stds_c, means_r, stds_r = zip(*data)

        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars_c = ax.bar(x - width/2, means_c, width, label="Chosen", color="#8cc5e3", edgecolor="black")
        bars_r = ax.bar(x + width/2, means_r, width, label="Rejected", color="#1a80bb", edgecolor="black")

        if plot_std:
            ax.errorbar(x - width/2, means_c, yerr=stds_c, fmt="none", ecolor="black", capsize=4)
            ax.errorbar(x + width/2, means_r, yerr=stds_r, fmt="none", ecolor="black", capsize=4)

        # significance stars
        for i, m in enumerate(categories):
            p = p_values.get(m, 1)
            if p < 0.05:
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                y = max(means_c[i], means_r[i]) + 0.002
                ax.text(x[i], y, stars, ha="center", va="bottom", fontsize=11)

        ax.set_ylabel("Correlation", fontsize=14)
        ax.set_title(f"Model attention vs human {feature_name}", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([smart_model_label(c) for c in categories], fontsize=13)
        ax.legend()
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)

        out_dir = os.path.join(path, "plots")
        os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{out_dir}/{tag}_{gaze_signal}_chosen_vs_rejected.png",
            dpi=300,
        )
        plt.show()

    # --------------------------------------------------
    # Reward-specific chosen / rejected
    # --------------------------------------------------
    @staticmethod
    def concatenate_reward_dfs(df1, df2):
        common = df1.columns.intersection(df2.columns)
        return pd.concat(
            [
                df1[common],
                df2[common].rename(columns={c: c + "_r" for c in common}),
            ],
            axis=1,
        )