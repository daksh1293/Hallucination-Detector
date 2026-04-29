import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

models = {
    "LLaMA 3.1 8B":    "data/llama3-8b_results.csv",
    "LLaMA 4 Scout":   "data/llama4-scout_results.csv",
    "LLaMA 3.3 70B":   "data/llama3-70b_results.csv",
}

domains = ["Science", "History", "Geography", "Technology"]
colors  = ["#3498db", "#2ecc71", "#e74c3c"]

dfs = {}
for model, path in models.items():
    if os.path.exists(path):
        dfs[model] = pd.read_csv(path)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("LLM Hallucination Analysis — TruthfulQA Benchmark",
             fontsize=14, fontweight="bold")

# Chart 1 — Overall hallucination rate
overall = [df["predicted_hallucination"].mean()*100 for df in dfs.values()]
bars = axes[0].bar(list(dfs.keys()), overall, color=colors)
axes[0].set_title("Overall Hallucination Rate (%)")
axes[0].set_ylabel("Hallucination Rate (%)")
axes[0].set_ylim(0, 35)
for bar, val in zip(bars, overall):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 val + 0.2, f"{val:.1f}%",
                 ha="center", fontweight="bold")

# Chart 2 — Domain-wise grouped bar
x = range(len(domains))
width = 0.25
for i, (model, df) in enumerate(dfs.items()):
    rates = [df[df["domain"]==d]["predicted_hallucination"].mean()*100
             for d in domains]
    offset = (i - 1) * width
    axes[1].bar([xi + offset for xi in x], rates,
                width, label=model, color=colors[i])
axes[1].set_title("Hallucination Rate by Domain")
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(domains)
axes[1].set_ylabel("Hallucination Rate (%)")
axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 20)

# Chart 3 — Avg support score
support = [df["support_score"].mean() for df in dfs.values()]
bars3 = axes[2].bar(list(dfs.keys()), support, color=colors)
axes[2].set_title("Average Support Score")
axes[2].set_ylabel("Support Score")
axes[2].set_ylim(0, 1)
for bar, val in zip(bars3, support):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 val + 0.01, f"{val:.3f}",
                 ha="center", fontweight="bold")

plt.tight_layout()
os.makedirs("data", exist_ok=True)
plt.savefig("data/model_comparison.png", dpi=150, bbox_inches="tight")
print("✅ Chart saved to data/model_comparison.png")
plt.show()