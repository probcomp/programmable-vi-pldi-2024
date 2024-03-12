#!/usr/bin/env python
# coding: utf-8

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

sns.set_theme(style="white")
rcParams["mathtext.fontset"] = "cm"
rcParams["font.family"] = "STIXGeneral"
rcParams["figure.autolayout"] = True
label_fontsize = 70  # Set the desired font size here


enum_air = pd.read_csv("./training_runs/genjax_air_enum_epochs_41.csv")
reinforce_air = pd.read_csv("./training_runs/genjax_air_reinforce_epochs_41.csv")
reinforce_iwae_air = pd.read_csv(
    "./training_runs/genjax_air_iwae_2_reinforce_epochs_41.csv"
)
mvd_air = pd.read_csv("./training_runs/genjax_air_mvd_epochs_41.csv")
mvd_iwae_air = pd.read_csv("./training_runs/genjax_air_iwae_2_mvd_epochs_41.csv")
hybrid_mvd_air = pd.read_csv(
    "./training_runs/genjax_air_hybrid_mvd_enum_epochs_41.csv",
)
hybrid_iwae_mvd_air = pd.read_csv(
    "./training_runs/genjax_air_iwae_2_hybrid_mvd_enum_epochs_41.csv",
)
rws_air_mvd = pd.read_csv("./training_runs/genjax_air_rws_10_mvd_epochs_6.csv")
rws_air_mvd_bs1 = pd.read_csv("./training_runs/genjax_air_rws_10_mvd_epochs_6_bs1.csv")
pyro_reinforce_air = pd.read_csv(
    "./training_runs/pyro_air_reinforce_epochs_41.csv",
)
pyro_reinforce_baselines_air = pd.read_csv(
    "./training_runs/pyro_air_reinforce_baselines_epochs_41.csv",
)

plt.rcParams["text.usetex"] = False
fig = plt.figure(figsize=(10, 12), dpi=240)
ax1 = fig.add_subplot(411)  # 3 rows, 1 column, 1st plot
ax2 = fig.add_subplot(412, sharex=ax1)  # 3 rows, 1 column, 2nd plot
ax3 = fig.add_subplot(413, sharex=ax1)
ax4 = fig.add_subplot(414)
fig.suptitle("Comparing algorithms for AIR", fontsize=label_fontsize / 2)
ax1.tick_params(axis="both", which="major", labelsize=label_fontsize / 3.2)
ax1.set_xscale("log")
ax2.tick_params(axis="both", which="major", labelsize=label_fontsize / 3.2)
ax2.set_xscale("log")
ax3.tick_params(axis="both", which="major", labelsize=label_fontsize / 3.2)
ax3.set_xscale("log")


def go_plot(
    ax,
    df,
    x,
    mean,
    std,
    label,
    cmap,
    color_idx,
    marker,
    fill=False,
    invert_for_pyro=False,
):
    if invert_for_pyro:
        scale = -1.0
    else:
        scale = 1.0
    l = ax.plot(
        np.array(df[x]),
        scale * np.array(df[mean]),
        label=label,
        color=cmap(color_idx),
        marker=marker,
    )
    if fill:
        ax.fill_between(
            np.array(df[x]),
            np.array(scale * df[mean]) - np.array(df[std]),
            np.array(scale * df[mean]) + np.array(df[std]),
            color=cmap(color_idx),
            alpha=0.2,
        )
    return l


ax1.set_ylim(250, 650)
num_lines = 7  # Number of lines you want to plot
cmap = plt.cm.get_cmap(
    "cividis", num_lines
)  # Replace 'viridis' with your chosen colormap

ra_l = go_plot(
    ax1,
    reinforce_air,
    "Epoch wall clock times",
    "ELBO loss",
    "Std ELBO loss",
    "Ours (REINFORCE)",
    cmap,
    0,
    "x",
)
ria_l = go_plot(
    ax1,
    reinforce_iwae_air,
    "Epoch wall clock times",
    "ELBO loss",
    "Std ELBO loss",
    "Ours (IWAE + REINFORCE)",
    cmap,
    1,
    "X",
)
enum_l = go_plot(
    ax1,
    enum_air,
    "Epoch wall clock times",
    "ELBO loss",
    "Std ELBO loss",
    "Ours (Enum)",
    cmap,
    2,
    "+",
)
mvd_l = go_plot(
    ax1,
    mvd_air,
    "Epoch wall clock times",
    "ELBO loss",
    "Std ELBO loss",
    "Ours (MVD)",
    cmap,
    3,
    "H",
)
mvd_hybrid_l = go_plot(
    ax1,
    hybrid_mvd_air,
    "Epoch wall clock times",
    "ELBO loss",
    "Std ELBO loss",
    "Ours (Hybrid: MVD x2, ENUM)",
    cmap,
    4,
    "P",
)
iwae_mvd_hybrid_l = go_plot(
    ax1,
    hybrid_iwae_mvd_air,
    "Epoch wall clock times",
    "ELBO loss",
    "Std ELBO loss",
    "Ours (Hybrid: (IWAE) MVD x2, ENUM)",
    cmap,
    5,
    "8",
)
mvd_iwae_l = go_plot(
    ax1,
    mvd_iwae_air,
    "Epoch wall clock times",
    "ELBO loss",
    "Std ELBO loss",
    "Ours (IWAE + MVD)",
    cmap,
    6,
    "d",
)

pyro_cmap = plt.cm.get_cmap("Reds", 8)  # Replace 'viridis' with your chosen colormap

pyro_r_l = go_plot(
    ax1,
    pyro_reinforce_air,
    "Epoch wall clock times",
    "ELBO loss",
    "Std ELBO loss",
    "Pyro (REINFORCE)",
    pyro_cmap,
    1,
    "v",
    fill=False,
    invert_for_pyro=True,
)
pyro_r_b_l = go_plot(
    ax1,
    pyro_reinforce_baselines_air,
    "Epoch wall clock times",
    "ELBO loss",
    "Std ELBO loss",
    "Pyro (REINFORCE + baselines)",
    pyro_cmap,
    2,
    "^",
    fill=False,
    invert_for_pyro=True,
)
ax1.set_ylabel("Objective", fontsize=label_fontsize / 3)


num_lines = 7  # Number of lines you want to plot
cmap = plt.cm.get_cmap(
    "cividis", num_lines
)  # Replace 'viridis' with your chosen colormap

go_plot(
    ax2,
    reinforce_air,
    "Epoch wall clock times",
    "Accuracy",
    "Std accuracy",
    "Ours (REINFORCE)",
    cmap,
    0,
    "x",
)
go_plot(
    ax2,
    reinforce_iwae_air,
    "Epoch wall clock times",
    "Accuracy",
    "Std accuracy",
    "Ours (IWAE + REINFORCE)",
    cmap,
    1,
    "X",
)
go_plot(
    ax2,
    enum_air,
    "Epoch wall clock times",
    "Accuracy",
    "Std accuracy",
    "Ours (Enum)",
    cmap,
    2,
    "+",
)
go_plot(
    ax2,
    mvd_air,
    "Epoch wall clock times",
    "Accuracy",
    "Std accuracy",
    "Ours (MVD)",
    cmap,
    3,
    "H",
)
go_plot(
    ax2,
    hybrid_mvd_air,
    "Epoch wall clock times",
    "Accuracy",
    "Std accuracy",
    "Ours (MVD)",
    cmap,
    4,
    "P",
)
go_plot(
    ax2,
    hybrid_iwae_mvd_air,
    "Epoch wall clock times",
    "Accuracy",
    "Std accuracy",
    "Ours (Hybrid: (IWAE) MVD x2, ENUM)",
    cmap,
    5,
    "8",
)
go_plot(
    ax2,
    mvd_iwae_air,
    "Epoch wall clock times",
    "Accuracy",
    "Std accuracy",
    "Ours (IWAE + MVD)",
    cmap,
    6,
    "d",
)

pyro_cmap = plt.cm.get_cmap("Reds", 8)  # Replace 'viridis' with your chosen colormap

go_plot(
    ax2,
    pyro_reinforce_air,
    "Epoch wall clock times",
    "Accuracy",
    "Std accuracy",
    "Pyro (REINFORCE)",
    pyro_cmap,
    1,
    "v",
    fill=False,
    invert_for_pyro=False,
)
go_plot(
    ax2,
    pyro_reinforce_baselines_air,
    "Epoch wall clock times",
    "Accuracy",
    "Std accuracy",
    "Pyro (REINFORCE + baselines)",
    pyro_cmap,
    2,
    "^",
    fill=False,
    invert_for_pyro=False,
)

ax2.set_ylabel("Accuracy", fontsize=label_fontsize / 3)
fig


# ## RWS


def go_plot_rws(ax, df, x, mean, label, cmap, color_idx, marker):
    return ax.plot(
        np.array(df[x]),
        np.array(df[mean]),
        label=label,
        color=cmap(color_idx),
        marker=marker,
    )


ax3.set_ylim(0.6, 1.0)
ax3.text(760, 0.67, "t ~ 16 mins", color="black", zorder=3, fontsize=label_fontsize / 4)

num_lines = 2
cmap = plt.cm.get_cmap("cividis", num_lines)

rws_air_l = go_plot_rws(
    ax3,
    rws_air_mvd_bs1,
    "Epoch wall clock times",
    "Accuracy",
    "Ours (batch size = 1, RWS(K = 10))",
    cmap,
    0,
    "x",
)

rws_air_l = go_plot_rws(
    ax3,
    rws_air_mvd,
    "Epoch wall clock times",
    "Accuracy",
    "Ours (batch size = 64, RWS(K = 10))",
    cmap,
    2,
    "X",
)

try:
    pyro_rws_air = pd.read_csv(
        "./training_runs/pyro_air_rws_epochs_6.csv",
    )
    pyro_cmap = plt.cm.get_cmap(
        "Reds", 8
    )  # Replace 'viridis' with your chosen colormap

    pyro_rws_l = go_plot_rws(
        ax3,
        pyro_rws_air,
        "Epoch wall clock times",
        "Accuracy",
        "Pyro (batch size = 1, RWS(K = 10))",
        pyro_cmap,
        2,
        "X",
    )
except:
    print("Pyro AIR RWS not found.")

ax3.set_xlabel("Time (s)", fontsize=label_fontsize / 3)
ax3.set_ylabel("Accuracy", fontsize=label_fontsize / 3)

handles, labels = [], []
for ax in [ax1, ax3]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

# Create a single legend on the last axis
ax4.legend(handles, labels, loc="upper center", ncol=2, fontsize=label_fontsize / 4.05)
ax4.set_frame_on(False)
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)
fig.tight_layout()
fig.savefig("./figs/fig_7_air_full.pdf", format="pdf")
