import pathlib
import pandas as pd

from models.plotter import Plotter
from models.compare_att import CompareAttention


# =========================================================
# Config
# =========================================================

ROOT = pathlib.Path(__file__).parent.resolve().parent.resolve()
RESULTS_PATH = ROOT / "oasstetc_data" / "attention" / "results"

LEVEL = "trials"   # trials | userset

FILES = {
    # "completed/correlation_" + LEVEL + "_fix_duration.csv": "TRT_f",
    # "completed/correlation_" + LEVEL + "_fix_duration_n.csv": "TRT_n_f",
    # "completed/correlation_" + LEVEL + "_first_fix_duration.csv": "FFD_f",
    "completed/correlation_" + LEVEL + "_first_fix_duration_n.csv": "FFD_n_f",
    # "completed/correlation_" + LEVEL + "_fix_number.csv": "nFix_f",
}


# =========================================================
# Load data
# =========================================================

dfs = {}

for file, gaze_signal in FILES.items():
    dfs[gaze_signal] = {}

    dfs[gaze_signal]["chosen"] = pd.read_csv(
        RESULTS_PATH / "chosen" / file,
        sep=";",
        index_col=0,
    ).dropna()

    dfs[gaze_signal]["rejected"] = pd.read_csv(
        RESULTS_PATH / "rejected" / file,
        sep=";",
        index_col=0,
    ).dropna()

    dfs[gaze_signal]["chosen_all"] = pd.read_csv(
        RESULTS_PATH
        / "chosen"
        / file.replace("correlation_trials_", "correlation_trials_alldata"),
        sep=";",
        index_col=0,
    )

    dfs[gaze_signal]["rejected_all"] = pd.read_csv(
        RESULTS_PATH
        / "rejected"
        / file.replace("correlation_trials_", "correlation_trials_alldata"),
        sep=";",
        index_col=0,
    )


# =========================================================
# Plot
# =========================================================

for gaze_signal, df in dfs.items():
    print(f"[PLOT] {gaze_signal}")

    p_values = CompareAttention.compute_posthoc_comparisons_correlation(
        df["chosen_all"], df["rejected_all"]
    )

    Plotter.plot_gaze_signal_chosenrejected(
        path=str(RESULTS_PATH) + "/",
        df=df,
        gaze_signal=gaze_signal,
        tag=LEVEL,
        plot_std=False,
        p_values=p_values,
    )