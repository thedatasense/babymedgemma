"""Generate the educational figures for the SAE and Jacobian-lens experiments.

    python figures.py     # writes figures/*.png from the result JSONs

Colorblind-safe palette (validated): flip/SAE = blue #0072B2, non-flip/PCA =
vermillion #D55E00, baseline = gray. Text stays in ink tokens, marks carry
identity, one axis per panel, legend + direct labels.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from babygemma.paths import ROOT as HERE
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#9a9a9a"
INK, SUB, GRID = "#222222", "#5f5f5f", "#e6e6e6"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": SUB,
    "xtick.color": SUB, "ytick.color": SUB, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load(p):
    return json.load(open(os.path.join(HERE, p)))


def _dlabel(ax, x, y, txt, dy=0.012, color=INK):
    ax.annotate(txt, (x, y), textcoords="offset points", xytext=(0, 6),
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=color)


# ---------------------------------------------------------------- SAE result
def fig_sae_alignment():
    d = load("results_gemma/sae_gemma/sae.json")
    sae = abs(d["sae_feature_max_cos_with_E_dir"]["cos"])
    pca = d["pca_top20_max_cos_with_E_dir"]
    rnd = d["random_dir_mean_cos_with_E_dir"]
    labels = ["SAE\nfeature", "PCA\n(top 20)", "Random\ndirection"]
    vals = [sae, pca, rnd]
    colors = [BLUE, VERM, GRAY]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    bars = ax.bar(labels, vals, color=colors, width=0.6, zorder=3)
    for b, v in zip(bars, vals):
        _dlabel(ax, b.get_x() + b.get_width() / 2, v, f"{v:.2f}")
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("|cosine| with the causal flip direction")
    ax.set_title("An unsupervised feature recovers the causal flip axis")
    ax.text(0.5, -0.24,
            "The rank-1 direction that experiment C uses to flip the answer is independently\n"
            "found by unsupervised decomposition: the SAE feature matches it (0.51), edging out\n"
            "PCA (0.48), far above a random direction (0.04). Not a supervised artifact.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.3, color=SUB)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIG, "sae_alignment.png"))
    plt.close(fig)


# -------------------------------------------------------------- jlens result
def fig_jlens_divergence():
    d = load("results_gemma/jlens_gemma/jlens.json")
    flip = d["lens_divergence_flip_clusters"]
    noflip = d["lens_divergence_nonflip_clusters"]
    layers = list(range(len(flip)))
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(layers, flip, "-o", color=BLUE, lw=2, ms=7, zorder=3, label="flipping clusters")
    ax.plot(layers, noflip, "-o", color=VERM, lw=2, ms=7, zorder=3, label="stable clusters")
    ax.annotate("flipping clusters", (layers[-1], flip[-1]), color=BLUE,
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=10.5, fontweight="bold")
    ax.annotate("stable clusters", (layers[-1], noflip[-1]), color=VERM,
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=10.5, fontweight="bold")
    ratio = flip[0] / (noflip[0] + 1e-9)
    ax.annotate(f"~{ratio:.0f}x apart\nfrom layer 0", (0.2, (flip[0] + noflip[0]) / 2),
                fontsize=10, color=INK, ha="left", va="center")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("within-cluster lens-margin divergence")
    ax.set_xlim(-0.3, len(flip) - 0.3 + 1.4)
    ax.set_ylim(0, max(flip) * 1.15)
    ax.set_xticks(layers)
    ax.set_title("The lens splits flipping from stable paraphrases, from the first layer")
    ax.text(0.5, -0.22,
            "The Jacobian lens reads the yes/no margin each layer is disposed to produce. Across\n"
            "paraphrases of a flipping question the readouts diverge ~9.5x more than for a stable\n"
            "question, already at layer 0 — corroborating the early causal locus from experiment C.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.3, color=SUB)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIG, "jlens_divergence.png"))
    plt.close(fig)


# ------------------------------------------------------------- jlens concept
def fig_jlens_concept():
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = list(range(7))
    yes = [0.0, 0.05, 0.15, 0.4, 0.75, 1.0, 1.1]
    no = [0.0, -0.04, -0.12, -0.35, -0.7, -0.95, -1.05]
    ax.plot(xs, yes, "-", color=BLUE, lw=2.4, zorder=3)
    ax.plot(xs, no, "-", color=VERM, lw=2.4, zorder=3)
    ax.axhline(0, color=SUB, lw=1, ls="--", zorder=1)
    ax.axvline(2.6, color=GRAY, lw=1.2, ls=":", zorder=1)
    ax.annotate("phrasing A  ->  “yes”", (6, 1.1), color=BLUE, fontweight="bold",
                fontsize=10.5, va="center", xytext=(4, 0), textcoords="offset points")
    ax.annotate("phrasing B  ->  “no”", (6, -1.05), color=VERM, fontweight="bold",
                fontsize=10.5, va="center", xytext=(4, 0), textcoords="offset points")
    ax.annotate("read together early,\nthen commit to opposite answers",
                (0.35, 0.7), fontsize=9.5, color=INK, ha="left")
    ax.set_xlabel("decoder layer (lens readout)")
    ax.set_ylabel("yes / no margin the layer is disposed to")
    ax.set_ylim(-1.4, 1.5)
    ax.set_xlim(0, 8.2)
    ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(["no", "0", "yes"])
    ax.set_title("What the Jacobian lens shows", pad=12)
    ax.text(0.5, -0.20,
            "Same image, same question, two phrasings. The lens reads the yes/no margin each layer\n"
            "is disposed to produce; where the two phrasings split is where the paraphrase flip commits.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.3, color=SUB)
    ax.grid(False)
    fig.savefig(os.path.join(FIG, "jlens_concept.png"))
    plt.close(fig)


# --------------------------------------------------------------- SAE concept
def _box(ax, x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=INK, zorder=3)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color=SUB, lw=1.6, zorder=1))


def fig_sae_concept():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 0.3, 1.9, 2.2, 1.2, "answer-position\nactivation", "#eef4fb", BLUE)
    _box(ax, 3.2, 1.9, 2.4, 1.2, "sparse\nautoencoder", "#f4f4f4", SUB)
    _box(ax, 6.3, 3.0, 2.0, 0.9, "feature 1", "#f7f7f7", GRAY)
    _box(ax, 6.3, 1.85, 2.0, 0.9, "feature k*", "#eef4fb", BLUE)
    _box(ax, 6.3, 0.7, 2.0, 0.9, "feature m", "#f7f7f7", GRAY)
    _box(ax, 9.3, 1.9, 2.4, 1.2, "causal flip\ndirection (exp. C)", "#fdece1", VERM)
    _arrow(ax, 2.5, 2.5, 3.2, 2.5)
    _arrow(ax, 5.6, 2.7, 6.3, 3.4); _arrow(ax, 5.6, 2.5, 6.3, 2.3); _arrow(ax, 5.6, 2.3, 6.3, 1.1)
    ax.add_patch(FancyArrowPatch((8.3, 2.3), (9.3, 2.5), arrowstyle="<->", mutation_scale=14,
                                 color=BLUE, lw=2, zorder=1))
    ax.text(8.8, 2.85, "cos 0.51", ha="center", color=BLUE, fontsize=10, fontweight="bold")
    ax.set_title("What the sparse autoencoder tests", loc="center")
    ax.text(6, 0.15, "It factors the residual stream into a few active features; we ask whether any "
            "feature's\ndirection matches the causal flip direction. One does (feature k*), unsupervised.",
            ha="center", va="bottom", fontsize=9.3, color=SUB)
    fig.savefig(os.path.join(FIG, "sae_concept.png"))
    plt.close(fig)


# --------------------------------------------------------------- NIH transfer
def fig_nih_transfer():
    n = load("results/nih_demo/nih_demo.json")
    j = load("results_gemma/jlens_gemma/jlens.json")
    native_acc, native_flip = 0.882, 0.082
    native_ratio = j["lens_divergence_flip_clusters"][0] / (j["lens_divergence_nonflip_clusters"][0] + 1e-9)
    nih_acc = n["nih_accuracy_zeroshot"]; nih_flip = n["nih_flip_rate"]
    nih_ratio = n["nih_flip_vs_nonflip_ratio"][0]
    panels = [
        ("Accuracy", native_acc, nih_acc, "competence\ndoes NOT transfer"),
        ("Flip rate", native_flip, nih_flip, "sensitivity\ntransfers"),
        ("Lens flip / stable\ndivergence ratio", native_ratio, nih_ratio, "mechanism\ntransfers (weaker)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.8))
    for ax, (title, nv, ni, note) in zip(axes, panels):
        bars = ax.bar(["native", "NIH\n(zero-shot)"], [nv, ni], color=[BLUE, VERM], width=0.6, zorder=3)
        for b, v in zip(bars, [nv, ni]):
            fmt = f"{v:.0%}" if v <= 1.0 else f"{v:.1f}x"
            ax.annotate(fmt, (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(nv, ni) * 1.28)
        ax.text(0.5, 0.92, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=SUB, style="italic")
        ax.set_axisbelow(True)
    fig.suptitle("Zero-shot on NIH ChestX-ray14: the mechanism transfers, competence does not",
                 fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "nih_transfer.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_sae_alignment()
    fig_sae_concept()
    fig_jlens_divergence()
    fig_jlens_concept()
    fig_nih_transfer()
    print("wrote:", sorted(os.listdir(FIG)))
