import os
import matplotlib.pyplot as plt


def plot_accuracy(results, title, save_path):
    names = list(results.keys())
    accs = list(results.values())

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, accs)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")

    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width()/2, a + 0.01, f"{a:.3f}",
                 ha="center", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()