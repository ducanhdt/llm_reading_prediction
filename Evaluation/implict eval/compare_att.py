import os
import re
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, ttest_rel
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap

from models.human_att import HumanAttentionExtractor
from models.model_att import ModelAttentionExtractor


class CompareAttention:
    """
    Canonical attention-vs-human comparison module.

    Responsibilities:
      - layer-wise Spearman correlation
      - aggregation across users / trials
      - chosen vs rejected statistical comparison
      - plotting utilities
    """

    def __init__(self, model_name, model_type, path):
        self.model_name = model_name
        self.model_type = model_type
        self.path = path

        self.gaze_features_names = {
            "fix_duration_n": "TRT",
            "fix_duration": "TRT",
            "first_fix_duration": "FFD",
            "first_fix_duration_n": "FFD",
            "fix_number": "nFix",
        }

    # =========================================================
    # Core statistics
    # =========================================================

    def compute_sc_layer(
        self,
        gaze_features_layer: dict,
        model_attention_layer: dict,
        gaze_feature="fix_duration_n",
        layer=0,
        filter_trials=None,
    ):
        if filter_trials is None:
            filter_trials = []

        scores = []
        for trial in gaze_features_layer:
            if float(trial) not in model_attention_layer:
                continue
            if filter_trials and float(trial) not in filter_trials:
                continue

            sc, _ = spearmanr(
                gaze_features_layer[trial][gaze_feature].values,
                model_attention_layer[float(trial)][layer]["attention"].values,
            )
            scores.append(sc)

        scores = sorted(scores)
        scores = scores[int(len(scores) * 0.05) :]  # trim bottom 5%

        return np.nanmean(scores), np.nanstd(scores), scores

    # =========================================================
    # User / Trial aggregation
    # =========================================================

    def compute_sc_user_set(
        self,
        user_set,
        gaze_feature="fix_duration_n",
        filter_completed=False,
        filter_cr=None,
        folder_attention="attention",
    ):
        att_dir = (
            f"{self.path}{folder_attention}/"
            f"{self.model_name.split('/')[-1]}/set_{user_set}/"
        )
        gaze_dir = f"{self.path}gaze_features_real/set_{user_set}/"

        gaze = HumanAttentionExtractor().load_gaze_features(gaze_dir)
        info = HumanAttentionExtractor().load_trials_info(gaze_dir)

        trials = info["complete"] if filter_completed else info["all"]

        if filter_cr == "chosen":
            trials = [t for t in trials if str(t).endswith(".1")]
        elif filter_cr == "rejected":
            trials = [t for t in trials if not str(t).endswith(".1")]

        att = ModelAttentionExtractor.load_attention_df(att_dir)

        sc_layers = {}
        for layer in list(next(iter(att.values())).keys()):
            mean_sc, _, _ = self.compute_sc_layer(
                gaze, att, gaze_feature, layer, trials
            )
            sc_layers[layer] = mean_sc

        return sc_layers

    def compute_sc_all_userset(
        self,
        gaze_feature="fix_duration_n",
        filter_completed=False,
        filter_cr=None,
        folder_attention="attention",
    ):
        all_att, all_gaze, filter_trials = {}, {}, []

        for user_set in range(1, 9):
            gaze_dir = f"{self.path}gaze_features_real/set_{user_set}/"
            att_dir = (
                f"{self.path}{folder_attention}/"
                f"{self.model_name.split('/')[-1]}/set_{user_set}/"
            )

            all_gaze |= HumanAttentionExtractor().load_gaze_features(gaze_dir)
            all_att |= ModelAttentionExtractor.load_attention_df(att_dir)

            info = HumanAttentionExtractor().load_trials_info(gaze_dir)
            trials = info["complete"] if filter_completed else info["all"]

            if filter_cr == "chosen":
                trials = [t for t in trials if str(t).endswith(".1")]
            elif filter_cr == "rejected":
                trials = [t for t in trials if not str(t).endswith(".1")]

            filter_trials.extend(trials)

        sc_layers, sc_layers_all = {}, {}
        for layer in list(next(iter(all_att.values())).keys()):
            mean_sc, std_sc, all_sc = self.compute_sc_layer(
                all_gaze, all_att, gaze_feature, layer, filter_trials
            )
            sc_layers[layer] = [mean_sc, std_sc]
            sc_layers_all[layer] = all_sc

        return sc_layers, sc_layers_all

    # =========================================================
    # Statistical tests
    # =========================================================

    @staticmethod
    def compute_posthoc_comparisons(df_chosen, df_rejected):
        t_stats, p_vals = [], []

        for model in df_chosen.columns:
            mask = ~df_chosen[model].isna() & ~df_rejected[model].isna()
            t, p = ttest_rel(df_chosen[model][mask], df_rejected[model][mask])
            t_stats.append(t)
            p_vals.append(p)

        _, p_corr, _, _ = multipletests(p_vals, method="fdr_tsbh")

        return pd.DataFrame(
            {
                "model": df_chosen.columns,
                "t_stat": t_stats,
                "p_uncorrected": p_vals,
                "p_corrected": p_corr,
            }
        )

    # =========================================================
    # Plotting
    # =========================================================

    def plot_attention_all_trials(
        self, gaze_features, attention_folder="attention", filter_completed=True
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"
        base = (
            f"{self.path}{attention_folder}/"
            f"{self.model_name.split('/')[-1]}/{folder_filter}/"
        )

        data = pd.DataFrame()
        for gf in gaze_features:
            df = pd.read_csv(
                f"{base}/correlation_trials_{gf}.csv", sep=";", index_col=0
            )[["mean"]]
            df.rename(columns={"mean": gf}, inplace=True)
            data = pd.concat([data, df], axis=1)

        self._plot_heatmap(
            data,
            title=self.model_name.split("/")[-1],
            out_name=f"{self.model_name.split('/')[-1]}_attention_layers_all_trials.png",
        )

    def plot_attention_chosenrejected(
        self, gaze_features, attention_folder="attention", filter_completed=True
    ):
        folder_filter = "completed" if filter_completed else "not_filtered"
        base = pathlib.Path(self.path) / attention_folder / self.model_name.split("/")[-1]

        for tag in ["chosen", "rejected"]:
            path = base / tag / folder_filter
            if not path.exists():
                continue

            data = pd.DataFrame()
            for gf in gaze_features:
                csv = path / f"correlation_trials_{gf}.csv"
                if not csv.exists():
                    continue
                df = pd.read_csv(csv, sep=";", index_col=0)[["mean"]]
                df.rename(columns={"mean": gf}, inplace=True)
                data = pd.concat([data, df], axis=1)

            self._plot_heatmap(
                data,
                title=f"{self.model_name.split('/')[-1]} ({tag})",
                out_name=f"{self.model_name.split('/')[-1]}_{tag}_attention_layers.png",
                subdir=tag,
            )

    # =========================================================
    # Internal plotting util
    # =========================================================

    def _plot_heatmap(self, data, title, out_name, subdir=""):
        if data.empty:
            return

        cmap = LinearSegmentedColormap.from_list(
            "blue_scale", ["#e0f2f9", "#8cc5e3", "#1a80bb", "#0a4f7d"]
        )

        plt.figure(figsize=(5, 8))
        sns.heatmap(data.values, cmap=cmap, linewidths=0.3, linecolor="white")

        n_layer = data.shape[0]
        plt.title(title, fontsize=16)
        plt.ylabel("Layer")
        plt.yticks(
            np.arange(0.5, n_layer),
            [f"L{n_layer-i}" for i in range(n_layer)],
            rotation=0,
        )
        plt.xticks(
            np.arange(0.5, data.shape[1]),
            [self.gaze_features_names[x] for x in data.columns],
            rotation=0,
        )

        out_dir = (
            pathlib.Path(self.path)
            / "attention"
            / "results"
            / "plots"
            / "attention_layers"
            / subdir
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / out_name, dpi=300, bbox_inches="tight")
        plt.close()