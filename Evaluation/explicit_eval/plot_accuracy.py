import os
import matplotlib.pyplot as plt


def plot_accuracy(results, title, save_dir="figures", filename=None):
    os.makedirs(save_dir, exist_ok=True)

    names = list(results.keys())
    accs = list(results.values())

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, accs)

    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)

    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=25)
    plt.tight_layout()

    if filename is None:
        filename = title.lower().replace(" ", "_") + ".png"

    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")