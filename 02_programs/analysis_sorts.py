# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
# Packages
from glob import glob
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import matplotlib
import seaborn as sns

import matplotlib.pyplot as plt


# %%
# ---- figure stuff:
cm = 1 / 2.54
width = 15.92 * cm
height = 10 * cm

matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.xmargin"] = 0.02
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.edgecolor"] = "0.15"
matplotlib.rcParams["axes.linewidth"] = 1.25
matplotlib.rcParams["figure.dpi"] = 800
matplotlib.rcParams["savefig.dpi"] = 800
matplotlib.rcParams["savefig.transparent"] = True
matplotlib.rcParams["savefig.format"] = "pdf"
matplotlib.rcParams["mathtext.fontset"] = "stixsans"


def sns_palette(n_colors):
    return sns.cubehelix_palette(n_colors=n_colors, start=0, rot=-0.05, dark=0.1, light=0.75)


# %%
# Read characteristics
char_files = glob("../04_results/option_sample/characteristics*.pq")
classifier = glob("../04_results/option_sample/classifier*.pq")

target = pd.concat(
    [pd.read_parquet(file, columns=["date", "optionid", "doi", "return_dh_daily_inv"]) for file in classifier]
)
target["date"] = target["date"].dt.to_timestamp()
target = target.rename(columns={"doi": "weight", "return_dh_daily_inv": "target"})
target = target.set_index(["date", "optionid"]).sort_index()


def read_files(file):
    data = pd.read_parquet(file)
    data = data[
        [
            c
            for c in data.columns
            if not c.startswith("ind")
            and c
            not in [
                "permno",
                "expiration_month",
                "C",
                "P",
                "bucket",
                "gain_dh_daily",
                "return_option",
                "return_dh_daily_s",
                "return_dh_daily_o",
                "return_dh_daily_margin",
                "margin_denominator",
                "date_col",
                "return_delevered",
                "leverage",
                "nopt",
                "return_dh_daily_inv",
            ]
        ]
    ]
    data = data.set_index("optionid", append=True)

    # create ranks
    data = (np.ceil(data.rank(pct=True, method="dense") * 100) - 1) // 10
    data[data.columns[data.nunique() < 10]] = np.nan
    return data


data = Parallel(n_jobs=cpu_count(), verbose=True)(delayed(read_files)(file) for file in char_files)
data = pd.concat(data)

# add target variable + weighting
data = data.merge(target, on=["date", "optionid"])
data["adjusted_target"] = data["target"] * data["weight"]


# start in 2003
data = data.sort_index()
data = data[data.index.get_level_values("date") >= "2003"]


# %%
# Loop over characteristics to get univariate returns/SRs
def create_premium(col):
    by_date = data.groupby(["date", col])["adjusted_target"].sum() / data.groupby(["date", col])["weight"].sum()
    if by_date.empty:
        return
    by_date = by_date.unstack()
    by_date["hml"] = by_date[9] - by_date[0]
    by_date = by_date.stack()
    by_date = by_date.to_frame(col)
    return by_date


loop_cols = [c for c in data.columns if c not in ["weight", "target", "adjusted_target"]]
premia = Parallel(n_jobs=cpu_count(), verbose=True)(delayed(create_premium)(col) for col in loop_cols)
premia = pd.concat(premia, axis=1)
premia.index.names = ["date", "q"]


# %%
# Do machine learning portfolios significantly beat univariate sorts?
from analysis_setup import prediction_dict, analysisLoc
from scipy.stats import ttest_ind

class_groups_with_info = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups_and_info.pq"),
    columns=["date", "optionid", "doi"],
)


factors = premia[premia.index.get_level_values("q") == "hml"].reset_index(level="q", drop=True)
factors = factors.sort_index()
factors = factors * np.where(factors.mean() < 0, -1, 1)
for model in ["L-En", "N-En"]:
    ports = prediction_dict[model]["predictions"].merge(class_groups_with_info, on=["date", "optionid"])
    ports = ports.groupby(["date", "port"]).apply(lambda x: (x.target * x.doi).sum() / x.doi.sum()).unstack()
    ports = ports[9] - ports[0]
    ports = ports.to_frame("model")
    ports.index = ports.index.to_timestamp()
    ports = ports.sort_index()

    outperformance = factors.apply(lambda x: ttest_ind(ports, x.dropna())[0], axis=0) > 1.96
    print(outperformance.T.squeeze().sort_values().head(20))


# %%
# HmL factor portfolios
factors = premia[premia.index.get_level_values("q") == "hml"].reset_index(level="q", drop=True)

mean_returns = factors.mean().abs()
SRs = factors.mean().abs() / factors.std()


fig, ax = plt.subplots(2, 1, figsize=(width, height), dpi=800)

sns.histplot(data=mean_returns, bins=15, ax=ax[0], color=sns_palette(4)[2], alpha=1)
ax[0].set_xlabel("Avg. Return")
ax[0].set_ylabel("")
ax[0].axvline(2.040 / 100, color="red", lw=2, ls=":")
ax[0].annotate("N-En", xy=(2.040 / 100 - 0.0017, 51), color="red", fontsize="x-large")
ax[0].axvline(1.303 / 100, color="gray", lw=2, ls=":")
ax[0].annotate("L-En", xy=(1.303 / 100 - 0.0016, 51), color="gray", fontsize="x-large")
ax[0].grid(axis="y", ls="--", color="gray", lw=1)
ax[0].set_axisbelow(True)

sns.histplot(data=SRs, bins=15, ax=ax[1], color=sns_palette(4)[0], alpha=1)
ax[1].set_xlabel("Realized SR")
ax[1].set_ylabel("")
ax[1].axvline(1.277, color="red", lw=2, ls=":")
ax[1].annotate("N-En", xy=(1.277 - 0.1, 43), color="red", fontsize="x-large")
ax[1].axvline(1.026, color="gray", lw=2, ls=":")
ax[1].annotate("L-En", xy=(1.026 - 0.1, 43), color="gray", fontsize="x-large")
ax[1].grid(axis="y", ls="--", color="gray", lw=1)
ax[1].set_axisbelow(True)

fig.tight_layout()
fig.savefig("../08_figures/univariate.pdf")


# %%
# Choose best-performing factor ex-post (maximum returns using univariate sorts with perfect foresight)
factors = premia[premia.index.get_level_values("q") == "hml"].reset_index(level="q", drop=True)
factors = factors * np.where(factors.mean() < 0, -1, 1)

max_returns = factors.max(axis=1).mean()
max_SR = max_returns / factors.max(axis=1).std()


returns = pd.Series({"Perfect foresight": max_returns * 100, "L-En": 1.303, "N-En": 2.040})
returns.index.name = "Model"
returns = returns.to_frame("value").reset_index()
returns["type"] = "Avg. Return"
SRs = pd.Series({"Perfect foresight": max_SR, "L-En": 1.026, "N-En": 1.277})
SRs.index.name = "Model"
SRs = SRs.to_frame("value").reset_index()
SRs["type"] = "Realized SR"
to_plot = pd.concat((returns, SRs))

fig, ax = plt.subplots(figsize=(width, height), dpi=800)
sns.barplot(data=to_plot, x="type", y="value", hue="Model", palette=sns_palette(3), ax=ax)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")
ax.grid(axis="y", ls="--", color="gray", lw=1)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_axisbelow(True)
ax.legend(title="")
fig.tight_layout()
fig.savefig("../08_figures/max_return.pdf")


# %%
