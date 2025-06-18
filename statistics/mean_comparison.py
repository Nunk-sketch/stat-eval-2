import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    "Category": ["RIDDLES", "FAQ", "CREATIVE PROMPTS", "STEM", "NON-STEM", "INTERDISCIPLINARY", "ACADEMIC"],
    "Mean_1.5": [0.616, 2.655, 0.780, 0.277, 0.147, 0.445, 0.234],
    "CI_low_1.5": [0.043, 1.565, 0.490, 0.191, 0.098, 0.252, 0.186],
    "CI_high_1.5": [1.188, 3.745, 1.069, 0.363, 0.195, 0.638, 0.292],
    "Mean_2.0": [0.431, 1.429, 1.434, 0.307, 0.287, 0.346, 0.302],
    "CI_low_2.0": [0.232, 0.829, 0.878, 0.245, 0.233, 0.231, 0.263],
    "CI_high_2.0": [0.629, 2.030, 1.989, 0.370, 0.341, 0.461, 0.341],
    "p_value": [0.0196, 0.4385, 0.1851, 0.0655, 7.3e-05, 0.4310, 0.00049]
}

df = pd.DataFrame(data)

def annotate_significance(p):
    if p is None:
        return ""
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

df["Significance"] = df["p_value"].apply(annotate_significance)

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(df))
width = 0.35

ax.bar(x - width/2, df["Mean_1.5"], width,
       yerr=[df["Mean_1.5"] - df["CI_low_1.5"], df["CI_high_1.5"] - df["Mean_1.5"]],
       capsize=5, label="Gemini 1.5", color="skyblue")

ax.bar(x + width/2, df["Mean_2.0"], width,
       yerr=[df["Mean_2.0"] - df["CI_low_2.0"], df["CI_high_2.0"] - df["Mean_2.0"]],
       capsize=5, label="Gemini 2.0", color="lightgreen")

for i, sig in enumerate(df["Significance"]):
    if sig:
        y_max = max(df["Mean_1.5"][i], df["Mean_2.0"][i])
        ax.text(i, y_max + 0.15, sig, ha='center', va='bottom', fontsize=14, color='red')

ax.set_xticks(x)
ax.set_xticklabels(df["Category"], rotation=30)
ax.set_ylabel("Proportion of English Words (%)")
ax.set_title("English Word Proportions by Content Category\n(Gemini 1.5 Flash vs Gemini 2.0 Flash)")
ax.legend()
plt.tight_layout()
plt.show()