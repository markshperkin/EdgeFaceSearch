import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

df = pd.read_csv("fitness.csv")
architectures = df["architecture"].unique()
learning_rates = sorted(df["learning_rate"].unique())

colors = list(mcolors.TABLEAU_COLORS.values())
color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}


plt.figure(figsize=(12, 8))

for arch in architectures:
    for lr in learning_rates:
        df_subset = df[(df["architecture"] == arch) & (df["learning_rate"] == lr)].sort_values("epoch")
        if df_subset.empty:
            continue
        plt.plot(
            df_subset["epoch"],
            df_subset["val_loss"],
            label=f"{arch}, lr={lr}",
            color=color_map[arch],
            marker="o",
        )

plt.xlabel("Epoch")
plt.ylabel("Latency")
plt.title("Latency over Epochs for Different Architectures")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
