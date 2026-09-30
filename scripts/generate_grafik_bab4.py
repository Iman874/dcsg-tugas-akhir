# -*- coding: utf-8 -*-
"""Generate grafik Bab 4 (perbandingan DCSG vs Word2Vec + tren parameter) sebagai PNG."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"<TA_ROOT>"
OUT = os.path.join(ROOT, "Evaluasi", "view", "grafik_bab4")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "Evaluasi", "result_comparison_full.csv"))
g = df[df["model"].str.endswith(":global")].copy()
g["dim"] = g["model"].str.extract(r"^dim(\d+)")[0].astype(int)
g["scheme"] = g["model"].str.extract(r"_(local_dominant|balanced|global_dominant):")[0]
g["k"] = g["model"].str.extract(r"_k(\d+)_")[0].astype(int)

w2v = df[df["model"].str.startswith("word2vec_")].copy()
w2v["dim"] = w2v["model"].str.extract(r"word2vec_dim(\d+)")[0].astype(int)

DIMS = [64, 128, 200]
BAR_COL_DCSG = "#2c7bb6"
BAR_COL_W2V = "#b0b0b0"

def grouped_metric(dcsg_vals, w2v_vals, ylabel, title, fname, ylim=None):
    x = np.arange(len(DIMS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, dcsg_vals, w, label="DCSG", color=BAR_COL_DCSG)
    ax.bar(x + w/2, w2v_vals, w, label="Word2Vec", color=BAR_COL_W2V)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}" for d in DIMS])
    ax.set_xlabel("Dimensi Embedding")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim: ax.set_ylim(ylim)
    # value labels
    for i, v in enumerate(dcsg_vals):
        ax.text(i - w/2, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(w2v_vals):
        ax.text(i + w/2, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close(fig)
    print("saved:", fname)

# 1. Semantic similarity (mean cosine)
d_ss = [g[g.dim==d]["semantic_similarity"].mean() for d in DIMS]
w_ss = [w2v[w2v.dim==d]["semantic_similarity"].mean() for d in DIMS]
grouped_metric(d_ss, w_ss, "Rata-rata Cosine Similarity",
               "Perbandingan Semantic Similarity DCSG vs Word2Vec",
               "g1_semantic_similarity.png", ylim=(0,1))

# 2. Word analogy top-5
d_a5 = [g[g.dim==d]["word_analogy_top5"].mean() for d in DIMS]
w_a5 = [w2v[w2v.dim==d]["word_analogy_top5"].mean() for d in DIMS]
grouped_metric(d_a5, w_a5, "Akurasi Top-5",
               "Perbandingan Word Analogy (Top-5) DCSG vs Word2Vec",
               "g2_word_analogy_top5.png", ylim=(0,0.5))

# 3. Sentiment accuracy
d_sa = [g[g.dim==d]["sentiment_accuracy"].mean() for d in DIMS]
w_sa = [w2v[w2v.dim==d]["sentiment_accuracy"].mean() for d in DIMS]
grouped_metric(d_sa, w_sa, "Akurasi",
               "Perbandingan Sentiment Classification DCSG vs Word2Vec",
               "g3_sentiment_accuracy.png", ylim=(0,0.8))

# 4. Tren parameter: pengaruh k terhadap cosine & akurasi (double axis)
ks = sorted(g["k"].unique())
k_sim = [g[g.k==k]["semantic_similarity"].mean() for k in ks]
k_acc = [g[g.k==k]["sentiment_accuracy"].mean() for k in ks]
fig, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(ks, k_sim, "o-", color=BAR_COL_DCSG, label="Cosine Similarity")
ax1.set_xlabel("Jumlah Negative Sample (k)")
ax1.set_ylabel("Cosine Similarity", color=BAR_COL_DCSG)
ax1.tick_params(axis="y", labelcolor=BAR_COL_DCSG)
ax2 = ax1.twinx()
ax2.plot(ks, k_acc, "s--", color="#f58518", label="Sentiment Accuracy")
ax2.set_ylabel("Sentiment Accuracy", color="#f58518")
ax2.tick_params(axis="y", labelcolor="#f58518")
ax1.set_title("Tren Pengaruh Negative Sample (k) terhadap Cosine dan Akurasi")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "g4_tren_k.png"), dpi=200)
plt.close(fig)
print("saved: g4_tren_k.png")

# 5. Tren parameter: skema bobot -> Spearman
schemes = ["local_dominant", "balanced", "global_dominant"]
s_spear = [g[g.scheme==s]["semantic_spearman"].mean() for s in schemes]
s_acc = [g[g.scheme==s]["sentiment_accuracy"].mean() for s in schemes]
x = np.arange(len(schemes)); w=0.35
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - w/2, s_spear, w, label="Spearman", color="#4c78a8")
ax.bar(x + w/2, s_acc, w, label="Sentiment Accuracy", color="#f58518")
ax.set_xticks(x)
ax.set_xticklabels(schemes, rotation=15)
ax.set_ylabel("Nilai Metrik")
ax.set_title("Tren Skema Bobot Kontribusi Konteks")
for i,v in enumerate(s_spear): ax.text(i-w/2, v+0.002, f"{v:.3f}", ha="center", fontsize=8)
for i,v in enumerate(s_acc): ax.text(i+w/2, v+0.002, f"{v:.3f}", ha="center", fontsize=8)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "g5_tren_scheme.png"), dpi=200)
plt.close(fig)
print("saved: g5_tren_scheme.png")

# 6. Waktu latih per dimensi (bar)
w2v_time = {"64": 1.40, "128": 1.69, "200": 2.06}
dcsg_time = {"64": 3.97, "128": 4.40, "200": 5.68}
x = np.arange(len(DIMS)); w=0.35
d_t = [dcsg_time[str(d)] for d in DIMS]
w_t = [w2v_time[str(d)] for d in DIMS]
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x-w/2, d_t, w, label="DCSG", color=BAR_COL_DCSG)
ax.bar(x+w/2, w_t, w, label="Word2Vec", color=BAR_COL_W2V)
ax.set_xticks(x); ax.set_xticklabels([str(d) for d in DIMS])
ax.set_xlabel("Dimensi Embedding"); ax.set_ylabel("Waktu Latih Rata-rata (jam)")
ax.set_title("Perbandingan Waktu Latih DCSG vs Word2Vec")
for i,v in enumerate(d_t): ax.text(i-w/2, v+0.03, f"{v:.2f}", ha="center", fontsize=9)
for i,v in enumerate(w_t): ax.text(i+w/2, v+0.03, f"{v:.2f}", ha="center", fontsize=9)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "g6_waktu_latih.png"), dpi=200)
plt.close(fig)
print("saved: g6_waktu_latih.png")

print("\nSELESAI. Output di:", OUT)
