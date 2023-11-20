# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
""" Single model analysis.

Looks in-depth at the predictions of a single model.

Includes the following analyses:
    # # Out of sample R2 analyses.
    # - Full sample R2 comparison
    # - Monthly R2 boxplot comparison
    # - Annual R2 boxplot comparison
    # - Diebold Mariano tests
    # - Forecast correlation

"""

# %%
# Setup
import os
import pandas as pd
import numpy as np
from scoring import ClarkWest, DieboldMariano, DieboldMariano_XS, ForecastCorrelation, XS_R2, TS_R2

from pylatex import Tabular, Math
from pylatex.utils import NoEscape

import matplotlib.pyplot as plt

# import matplotlib
import seaborn as sns

from analysis_setup import (
    prediction_dict,
    sns_palette,
    width,
    height,
    group_dict,
    analysisLoc,
    # n_top_features,
    # n_bot_features,
)

class_groups = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups.pq"), columns=["date", "optionid", "type", "bucket"]
)
class_groups_with_info = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups_and_info.pq"),
    columns=["date", "optionid", "doi"],
)


# %% OOS R2 full sample
to_plot = []
for i, model in enumerate(prediction_dict):
    if i == 0:
        target = prediction_dict[model]["predictions"]["target"].copy()
    tmp = prediction_dict[model]["predictions"]["predicted"]
    tmp.name = model
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot = pd.concat((to_plot, target), axis=1)
cw_test = ClarkWest(to_plot, 12, benchmark_type="zero", cw_adjust=False)

stars = []
for c in cw_test:
    if c > 3.09:
        stars.append("***")
    elif c > 2.326:
        stars.append("**")
    elif c > 1.645:
        stars.append("*")
    else:
        stars.append("")
stars = pd.Series(data=stars, index=cw_test.index)

to_plot = []
for model in prediction_dict:
    tmp = prediction_dict[model]["class_scores"]["type"].copy()
    tmp.loc["all"] = prediction_dict[model]["scores"].values
    tmp = tmp.reset_index()
    tmp["model"] = model.split("_")[0]
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot)
to_plot["type"] = to_plot["type"].replace({"all": "All", "call": "Call", "put": "Put"})


fig, axes = plt.subplots(figsize=(width, height), dpi=1000)
sns.barplot(x="model", y="score", hue="type", data=to_plot, ax=axes, palette=sns_palette(3))
axes.set_ylabel("")
axes.set_xlabel("")
axes.legend(frameon=False, title=None)
axes.axhline(0, color="k", ls="--")
fig.tight_layout()
axes.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
axes.set_axisbelow(True)
for i, (_, txt) in enumerate(stars.iteritems()):
    axes.annotate(txt, (i, -0.009), ha="center")
fig.savefig("../08_figures/r2_comparison.pdf", bbox_inches="tight")

# %% OOS R2 full sample ---- CROSS-SECTIONAL VERSION FOLLOWING
# Rapach et al. "Firm Characteristics and Expected Returns", 2021
# Addresses Weak vs. Strong factors to soME Degree.
to_plot = []
for i, model in enumerate(prediction_dict):
    if i == 0:
        target = prediction_dict[model]["predictions"][["optionid", "target"]].copy()
    tmp = prediction_dict[model]["predictions"]["predicted"].copy()
    tmp.name = model
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot = pd.concat((to_plot, target), axis=1)

# add type (call/put) information:
to_plot = to_plot.reset_index()
to_plot = to_plot.merge(class_groups[["date", "optionid", "type"]], on=["date", "optionid"])
to_plot = to_plot.drop(columns=["optionid"])
to_plot = to_plot.set_index("date")


calls = XS_R2(to_plot[to_plot["type"] == "call"].drop(columns=["type"]))
calls = calls.to_frame()
calls["type"] = "Call"
puts = XS_R2(to_plot[to_plot["type"] == "put"].drop(columns=["type"]))
puts = puts.to_frame()
puts["type"] = "Put"
xs_r2 = XS_R2(to_plot.drop(columns=["type"]))
xs_r2 = xs_r2.to_frame()
xs_r2["type"] = "All"
xs_r2 = pd.concat((calls, puts, xs_r2), axis=0)
xs_r2 = xs_r2.reset_index()
xs_r2.columns = ["model", "score", "type"]
# xs_r2["type"] = r"$R^2_{OS;XS}$"


cw_test = ClarkWest(to_plot.drop(columns=["type"]), 12, benchmark_type="xs", cw_adjust=False)
stars = []
for c in cw_test:
    if c > 3.09:
        stars.append("***")
    elif c > 2.326:
        stars.append("**")
    elif c > 1.645:
        stars.append("*")
    else:
        stars.append("")
stars = pd.Series(data=stars, index=cw_test.index)


fig, axes = plt.subplots(figsize=(width, height), dpi=1000)
sns.barplot(x="model", y="score", hue="type", data=xs_r2, ax=axes, palette=sns_palette(3))
axes.set_ylabel("")
axes.set_xlabel("")
axes.legend(frameon=False, title=None)
axes.axhline(0, color="k", ls="--")
fig.tight_layout()
axes.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
axes.set_axisbelow(True)
for i, (_, txt) in enumerate(stars.iteritems()):
    axes.annotate(txt, (i, -0.012), ha="center")
fig.savefig("../08_figures/xs_r2_comparison.pdf", bbox_inches="tight")

# %% OOS R2 per year, boxplot
to_plot = []
for model in prediction_dict:
    tmp = prediction_dict[model]["annual_scores"]["predicted"].copy()
    tmp.name = "score"
    tmp = tmp.reset_index()
    tmp["model"] = model.split("_")[0]
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot)

fig, axes = plt.subplots(2, 1, figsize=(width, height * 1.2), dpi=1000, sharex=True, sharey=True)
sns.boxplot(
    x="model",
    y="score",
    data=to_plot,
    ax=axes[0],
    palette=sns.color_palette("hls", 11),
    showfliers=False,
    showmeans=True,
    whis=[2.5, 97.5],
    meanprops={
        "marker": "o",
        "markeredgecolor": "black",
        "markerfacecolor": "white",
        "markersize": "6",
    },
    boxprops={"edgecolor": (0, 0, 0)},
    medianprops={"color": (0, 0, 0)},
    capprops={"color": (0, 0, 0)},
    whiskerprops={"color": (0, 0, 0)},
    linewidth=1,
)
axes[0].axhline(0, color="k", ls="--", lw=1)
# axes[0].grid(ls="--", axis="y", color="black", lw=0.5)
axes[0].set_xlabel("")
axes[0].set_ylabel("$R_{OS}^2$")
# fig.savefig("../08_figures/annual_r2_boxplot.pdf")


# ---- XS OS R2 annual plot:
to_plot = []
for i, model in enumerate(prediction_dict):
    if i == 0:
        target = prediction_dict[model]["predictions"]["target"].copy()
    tmp = prediction_dict[model]["predictions"]["predicted"]
    tmp.name = model
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot = pd.concat((to_plot, target), axis=1)
xs_r2 = to_plot.groupby(to_plot.index.year).apply(lambda x: XS_R2(x))
xs_r2 = xs_r2.stack().reset_index()
xs_r2.columns = ["date", "model", "score"]

sns.boxplot(
    x="model",
    y="score",
    data=xs_r2,
    ax=axes[1],
    palette=sns.color_palette("hls", 11),
    showfliers=False,
    showmeans=True,
    whis=[2.5, 97.5],
    meanprops={
        "marker": "o",
        "markeredgecolor": "black",
        "markerfacecolor": "white",
        "markersize": "6",
    },
    boxprops={"edgecolor": (0, 0, 0)},
    medianprops={"color": (0, 0, 0)},
    capprops={"color": (0, 0, 0)},
    whiskerprops={"color": (0, 0, 0)},
    linewidth=1,
)
axes[1].axhline(0, color="k", ls="--", lw=1)
axes[1].set_xlabel("")
# axes[1].grid(ls="--", axis="y", color="black", lw=0.5)
axes[1].set_ylabel("$R_{OS;XS}^2$")
fig.tight_layout()
fig.subplots_adjust(hspace=0.05)
fig.savefig("../08_figures/annual_r2_boxplot.pdf")


# %% Diebold Mariano test
to_plot = []
for i, model in enumerate(prediction_dict):
    if i == 0:
        target = prediction_dict[model]["predictions"]["target"].copy()
    tmp = prediction_dict[model]["predictions"]["predicted"]
    tmp.name = model
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot = pd.concat((to_plot, target), axis=1)

diebold_test = DieboldMariano(to_plot, 12)
diebold_test_XS = DieboldMariano_XS(to_plot, 12)


def DM_latex_table(data, save_name: str, skip_index=False, num_format="%.4f"):
    """Generates latex table from pd.DataFrame

    Args:
        data (pd.DataFrame): input data
        save_name (str): save name of table
        skip_index (bool, optional): Whether to skip index when creating table. Defaults to False.
        num_format (str, optional): Format for numbers.
    """

    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    if skip_index:
        table = Tabular("".join(["c"] * data.shape[1]), booktabs=True)
    else:
        table = Tabular("".join(["l"] + ["c"] * data.shape[1]), booktabs=True)

    if skip_index:
        table.add_row(data.columns.tolist())
        table.add_hline()
    else:
        table.add_row([""] + data.columns.tolist())
        table.add_hline(start=2, end=len(data.columns) + 1)

    for idx, row in data.iterrows():
        to_add = []
        for num in row:
            if np.isnan(num):
                to_add.append("")
            else:
                if np.abs(num) > 2.56:
                    num = num_format % num
                    to_add.append(math(r"\textcolor{Cyan}{\textbf{%s}}" % num))
                elif np.abs(num) > 1.96:
                    num = num_format % num
                    to_add.append(math(r"\textcolor{Blue}{\textbf{%s}}" % num))
                else:
                    to_add.append(math(num_format % num))
        table.add_row([idx] + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


DM_latex_table(diebold_test, "dm", skip_index=False, num_format="%.2f")
DM_latex_table(diebold_test_XS, "dm_xs", skip_index=False, num_format="%.2f")


# %% Forecast correlation table:
to_plot = []
for model in prediction_dict:
    tmp = prediction_dict[model]["predictions"]["predicted"].copy()
    tmp.name = model
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)

forecast_correlation = ForecastCorrelation(to_plot)


def latex_table(data, save_name: str, skip_index=False, num_format="%.4f"):
    """Generates latex table from pd.DataFrame

    Args:
        data (pd.DataFrame): input data
        save_name (str): save name of table
        skip_index (bool, optional): Whether to skip index when creating table. Defaults to False.
        num_format (str, optional): Format for numbers.
    """

    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    if skip_index:
        table = Tabular("".join(["c"] * data.shape[1]), booktabs=True)
    else:
        table = Tabular("".join(["l"] + ["c"] * data.shape[1]), booktabs=True)

    if skip_index:
        table.add_row(data.columns.tolist())
        table.add_hline()
    else:
        table.add_row([""] + data.columns.tolist())
        table.add_hline(start=2, end=len(data.columns) + 1)

    for idx, row in data.iterrows():
        to_add = []
        for num in row:
            if np.isnan(num):
                to_add.append("")
            else:
                if num > 0.9:
                    num = num_format % num
                    to_add.append(math(r"\textcolor{Cyan}{\textbf{%s}}" % num))
                elif num > 0.7:
                    num = num_format % num
                    to_add.append(math(r"\textcolor{Blue}{\textbf{%s}}" % num))
                else:
                    to_add.append(math(num_format % num))
        table.add_row([idx] + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


latex_table(forecast_correlation, "forecast_correlation", skip_index=False, num_format="%.2f")

# %% Linear vs. nonlinear performance over time ---- XS R2
fig, axes = plt.subplots(1, 2, figsize=(width, height * 0.8), dpi=1000)
to_plot = []
for i, model in enumerate(["L-En", "N-En"]):
    if i == 0:
        target = prediction_dict[model]["predictions"]["target"].copy()
    tmp = prediction_dict[model]["predictions"]["predicted"]
    tmp.name = model
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot = pd.concat((to_plot, target), axis=1)
xs_r2 = to_plot.groupby("date").apply(lambda x: XS_R2(x))
xs_r2 = xs_r2.stack().reset_index()
xs_r2.columns = ["date", "model", "score"]
xs_r2["date"] = xs_r2["date"].dt.to_timestamp()

sns.scatterplot(x="date", y="score", hue="model", data=xs_r2, ax=axes[0], palette=sns_palette(2))
axes[0].set_xlabel("")
axes[0].set_ylabel("$R^2_{OS;XS}$")
axes[0].axhline(0, color="k", ls="--")
axes[0].legend(frameon=False, fontsize=7)
fig.autofmt_xdate()
for tick in axes[0].get_xticklabels():
    tick.set_rotation(45)

xs_r2 = xs_r2.set_index(["date", "model"]).unstack()
xs_r2.columns = ["linear", "nonlinear"]

sns.scatterplot(x="linear", y="nonlinear", data=xs_r2, ax=axes[1], color=(0.2, 0.2, 0.2, 0.8), edgecolor="white")
axes[1].set_xlabel("L-En")
axes[1].set_ylabel("N-En")
axes[1].axline([0, 0], slope=1, color="black", ls="--", lw=0.5)
axes[1].set_xlim(axes[1].get_xlim())
axes[1].set_ylim(axes[1].get_ylim())
axes[1].set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
axes[1].set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
for tick in axes[1].get_xticklabels():
    tick.set_rotation(0)
axes[1].fill_between(
    axes[1].get_xlim(),
    (
        max(axes[1].get_ylim()[0], axes[1].get_xlim()[0]),
        min(axes[1].get_ylim()[1], axes[1].get_xlim()[1]),
    ),
    (axes[1].get_ylim()[1], axes[1].get_ylim()[1]),
    color=(0, 1, 0, 0.1),
)
axes[1].fill_between(
    axes[1].get_xlim(),
    (axes[1].get_ylim()[0], axes[1].get_ylim()[0]),
    (
        max(axes[1].get_ylim()[0], axes[1].get_xlim()[0]),
        min(axes[1].get_ylim()[1], axes[1].get_xlim()[1]),
    ),
    color=(1, 0, 0, 0.1),
)
sns.scatterplot(x="linear", y="nonlinear", data=xs_r2.loc["2019-12":"2020-12"], ax=axes[1], color="red")
fig.tight_layout()
fig.subplots_adjust(wspace=0.15)
fig.savefig("../08_figures/monthly_xs_r2_linear_vs_nonlinear.pdf")


# %% Linear vs. nonlinear performance over time
fig, axes = plt.subplots(1, 2, figsize=(width, height * 0.8), dpi=1000)

to_plot = []
for model in ["L-En", "N-En"]:
    tmp = prediction_dict[model]["monthly_scores"]["predicted"].copy()
    tmp.name = "score"
    tmp = tmp.reset_index()
    tmp["model"] = model.split("_")[0]
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot)
to_plot["date"] = to_plot["date"].dt.to_timestamp()
to_plot = to_plot.set_index("date")

sns.scatterplot(x="date", y="score", hue="model", data=to_plot, ax=axes[0], palette=sns_palette(2))
axes[0].set_xlabel("")
axes[0].set_ylabel("$R^2_{OS}$")
axes[0].axhline(0, color="k", ls="--")
axes[0].legend(frameon=False, fontsize=7)
fig.autofmt_xdate()
for tick in axes[0].get_xticklabels():
    tick.set_rotation(45)

to_plot = []
for model in ["L-En", "N-En"]:
    tmp = prediction_dict[model]["monthly_scores"]["predicted"].copy()
    tmp.name = "score"
    tmp = tmp.reset_index()
    tmp["model"] = model.split("_")[0]
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot)
to_plot = to_plot.set_index(["date", "model"], append=True).unstack()
to_plot = to_plot.reset_index(level=0, drop=True)
to_plot.columns = ["linear", "nonlinear"]

sns.scatterplot(
    x="linear",
    y="nonlinear",
    data=to_plot,
    ax=axes[1],
    color=(0.2, 0.2, 0.2, 0.8),
    edgecolor="white",
)
axes[1].set_xlabel("L-En")
axes[1].set_ylabel("N-En")
axes[1].axline([0, 0], slope=1, color="black", ls="--", lw=0.5)
axes[1].set_xlim(axes[1].get_xlim())
axes[1].set_ylim(axes[1].get_xlim())
axes[1].set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
axes[1].set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
for tick in axes[1].get_xticklabels():
    tick.set_rotation(0)
axes[1].fill_between(
    axes[1].get_xlim(),
    (
        max(axes[1].get_ylim()[0], axes[1].get_xlim()[0]),
        min(axes[1].get_ylim()[1], axes[1].get_xlim()[1]),
    ),
    (axes[1].get_ylim()[1], axes[1].get_ylim()[1]),
    color=(0, 1, 0, 0.1),
)
axes[1].fill_between(
    axes[1].get_xlim(),
    (axes[1].get_ylim()[0], axes[1].get_ylim()[0]),
    (
        max(axes[1].get_ylim()[0], axes[1].get_xlim()[0]),
        min(axes[1].get_ylim()[1], axes[1].get_xlim()[1]),
    ),
    color=(1, 0, 0, 0.1),
)
sns.scatterplot(x="linear", y="nonlinear", data=to_plot.loc["2019-12":"2020-12"], ax=axes[1], color="red")
fig.tight_layout()
fig.subplots_adjust(wspace=0.15)
fig.savefig("../08_figures/monthly_r2_linear_vs_nonlinear.pdf")


# %% Ensemble SHAP importance
to_plot = []
for model in ["N-En"]:
    tmp = prediction_dict[model]["SHAPs"]["full"].copy()
    N = tmp.N
    tmp = tmp[[c for c in tmp.columns if c.startswith("m_")]]
    tmp = tmp[
        [
            c
            for c in tmp.columns
            if c
            not in [
                "m_information-options",
                "m_information-underlying",
                "m_instrument-bucket;instrument-contract",
                "m_degree",  # lol.
            ]
        ]
    ]
    tmp = tmp.multiply(N, axis=0)  # scale to sum of SHAPs
    means = tmp.sum()
    means = means.divide(means.sum())
    means = means.sort_values(ascending=False)
    tmp = tmp.divide(tmp.sum(axis=1), axis=0)  # relative
    tmp = tmp[means.index]
    tmp.columns = [t.split("m_")[-1] for t in tmp.columns]
    tmp.columns = [group_dict[c] for c in tmp.columns]
    means.index = tmp.columns
    tmp = tmp.stack().to_frame().reset_index()
    tmp.columns = ["date", "Feature", "SHAP"]
    tmp = tmp.set_index("Feature")
    tmp = tmp.loc[means.index]
    tmp = tmp.reset_index()

    fig, ax = plt.subplots(figsize=(width, height), dpi=1000)
    sns.stripplot(
        x="Feature",
        y="SHAP",
        data=tmp,
        ax=ax,
        palette=sns.color_palette("hls", 12),
        alpha=1,
        orient="v",
    )
    sns.barplot(
        x="Feature",
        y="SHAP",
        data=tmp,
        ax=ax,
        palette=sns.color_palette("hls", 12),
        alpha=0.5,
        ci=None,
        orient="v",
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    # plt.xticks(rotation=0, ha="right")
    fig.tight_layout()
    fig.savefig("../08_figures/shaps_ensembles.pdf")


# %% Ensemble SHAP importance per bucket
rename_dict = {
    "short_term_atm": "short term atm",
    "long_term_atm": "long term atm",
    "short_term_itm_call": "short term itm C",
    "long_term_itm_call": "long term itm C",
    "short_term_itm_put": "short term itm P",
    "long_term_itm_put": "long term itm P",
    "short_term_otm_call": "short term otm C",
    "long_term_otm_call": "long term otm C",
    "short_term_otm_put": "short term otm P",
    "long_term_otm_put": "long term otm P",
}

for model in ["N-En"]:

    full = prediction_dict[model]["SHAPs"]["full"].copy()

    N = full.N
    full = full[[c for c in full.columns if c.startswith("m_")]]
    full = full[
        [
            c
            for c in full.columns
            if c
            not in [
                "m_information-options",
                "m_information-underlying",
                "m_instrument-bucket;instrument-contract",
                "m_degree",
            ]
        ]
    ]
    full = full.multiply(N, axis=0)  # scale to sum of SHAPs
    means = full.sum()
    means = means.divide(means.sum())
    means = means.sort_values(ascending=False)
    full = full.divide(full.sum(axis=1), axis=0)  # relative
    full = full[means.index]
    full.columns = [t.split("m_")[-1] for t in full.columns]
    full.columns = [group_dict[c] for c in full.columns]
    means.index = full.columns
    full = full.stack().to_frame().reset_index()
    full.columns = ["date", "Feature", "SHAP"]
    full = full.set_index("Feature")
    full = full.loc[means.index]
    full = full.reset_index()

    order = full.groupby("Feature").SHAP.mean().sort_values(ascending=False).index.get_level_values("Feature")

    tmp = prediction_dict[model]["SHAPs"]["bucket"].copy()
    tmp.reset_index("bucket", inplace=True)
    to_plot = []
    for idx, ttemp in tmp.groupby("bucket"):
        N = ttemp.N
        ttemp = ttemp[[c for c in ttemp.columns if c.startswith("m_")]]
        ttemp = ttemp[
            [
                c
                for c in ttemp.columns
                if c
                not in [
                    "m_information-options",
                    "m_information-underlying",
                    "m_instrument-bucket;instrument-contract",
                    "m_degree",
                ]
            ]
        ]
        ttemp = ttemp.multiply(N, axis=0)  # scale to sum of SHAPs
        means = ttemp.sum()
        means = means.divide(means.sum())
        means = means.sort_values(ascending=False)
        ttemp = ttemp.divide(ttemp.sum(axis=1), axis=0)  # relative
        ttemp = ttemp[means.index]
        ttemp.columns = [t.split("m_")[-1] for t in ttemp.columns]
        ttemp.columns = [group_dict[c] for c in ttemp.columns]
        means.index = ttemp.columns
        ttemp = ttemp.stack().to_frame().reset_index()
        ttemp.columns = ["date", "Feature", "SHAP"]
        ttemp = ttemp.set_index("Feature")
        ttemp = ttemp.loc[means.index]
        ttemp = ttemp.reset_index()
        ttemp["bucket"] = idx
        # ttemp.Feature = ttemp.Feature.astype("category")
        # ttemp.Feature.cat.set_categories(order, inplace = True)
        to_plot.append(ttemp)

    to_plot = pd.concat(to_plot)
    to_plot.set_index("bucket", inplace=True)
    to_plot = to_plot[~to_plot.index.str.contains("nan")]

    do_one_page = True
    if do_one_page:
        fig, ax = plt.subplots(5, 2, figsize=(width, 15 / 2.54), sharex=False)

        for i, bucket in enumerate(to_plot.index.get_level_values("bucket").unique()):
            if "long" in bucket:
                j = 1
            else:
                j = 0
                i -= 5

            tmp = to_plot[to_plot.index.get_level_values("bucket") == bucket].copy()
            tmp.reset_index(drop=True, inplace=True)
            tmp.set_index("date", inplace=True)
            order = tmp.groupby("Feature").mean().sort_values("SHAP", ascending=False).index.values

            sns.stripplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                ax=ax[i, j],
                order=order,
                palette=sns.color_palette("hls", 12),
                alpha=1,
                orient="v",
                zorder=1,
            )

            sns.barplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                ax=ax[i, j],
                order=order,
                palette=sns.color_palette("hls", 12),
                alpha=0.5,
                ci=None,
                orient="v",
                zorder=0,
            )
            ax[i, j].set_ylabel("")
            ax[i, j].set_xlabel("")
            ax[i, j].tick_params(axis="x", rotation=90, direction="in", pad=-5, labelbottom=True, zorder=2, size=4)
            ax[i, j].set_title(rename_dict[bucket], fontsize=8)
            plt.setp(ax[i, j].get_xticklabels(), va="bottom")

        # for label in ax.get_xticklabels():
        # label.set_va("bottom")

        fig.tight_layout()
        fig.savefig("../08_figures/shaps_ensembles_buckets.pdf")

    else:

        second_page = ["long_term_otm_call", "long_term_otm_put", "short_term_otm_call", "short_term_otm_put"]

        to_plot_2 = to_plot[to_plot.index.get_level_values("bucket").isin(second_page)]
        to_plot_1 = to_plot[~to_plot.index.get_level_values("bucket").isin(second_page)]

        fig, ax = plt.subplots(3, 2, figsize=(width, 15 / 2.54), sharex=True)

        for i, bucket in enumerate(to_plot_1.index.get_level_values("bucket").unique()):

            if "long" in bucket:
                j = 1
            else:
                j = 0
                i -= 3

            tmp = to_plot_1[to_plot_1.index.get_level_values("bucket") == bucket].copy()
            tmp.reset_index(drop=True, inplace=True)
            # tmp.sort_values(["Feature", "date"], inplace = True)
            tmp.set_index("date", inplace=True)

            sns.stripplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                ax=ax[i, j],
                order=order,
                palette=sns.color_palette("hls", 12),
                alpha=1,
                orient="v",
            )
            sns.barplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                ax=ax[i, j],
                order=order,
                palette=sns.color_palette("hls", 12),
                alpha=0.5,
                ci=None,
                orient="v",
            )
            ax[i, j].set_ylabel("")
            ax[i, j].set_xlabel("")
            ax[i, j].tick_params(axis="x", rotation=90)
            ax[i, j].set_title(rename_dict[bucket], fontsize=8)
        fig.tight_layout()
        fig.savefig("../08_figures/shaps_ensembles_buckets_1.pdf")

        fig, ax = plt.subplots(2, 2, figsize=(width, 10 / 2.54), sharex=True)
        for i, bucket in enumerate(to_plot_2.index.get_level_values("bucket").unique()):

            if "long" in bucket:
                j = 1
            else:
                j = 0
                i -= 2

            tmp = to_plot_2[to_plot_2.index.get_level_values("bucket") == bucket].copy()
            tmp.reset_index(drop=True, inplace=True)
            # tmp.sort_values(["Feature", "date"], inplace = True)
            tmp.set_index("date", inplace=True)
            sns.stripplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                order=order,
                ax=ax[i, j],
                palette=sns.color_palette("hls", 12),
                alpha=1,
                orient="v",
            )
            sns.barplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                order=order,
                ax=ax[i, j],
                palette=sns.color_palette("hls", 12),
                alpha=0.5,
                ci=None,
                orient="v",
            )
            ax[i, j].set_ylabel("")
            ax[i, j].set_xlabel("")
            ax[i, j].tick_params(axis="x", rotation=90)
            ax[i, j].set_title(rename_dict[bucket], fontsize=8)

        fig.tight_layout()
        fig.savefig("../08_figures/shaps_ensembles_buckets_2.pdf")

# %% Feature group importance for ensembles over time (heatmap):
to_plot = []
for model in ["N-En"]:
    tmp = prediction_dict[model]["SHAPs"]["full"].copy()
    N = tmp.N
    tmp = tmp[[c for c in tmp.columns if c.startswith("m_")]]
    tmp = tmp[
        [
            c
            for c in tmp.columns
            if c
            not in [
                "m_information-options",
                "m_information-underlying",
                "m_instrument-bucket;instrument-contract",
                "m_degree",
            ]
        ]
    ]
    tmp = tmp.multiply(N, axis=0)  # scale to sum of SHAPs
    tmp = tmp.groupby(tmp.index.year).mean()
    tmp = tmp.divide(tmp.sum(axis=1), axis=0)  # relative
    tmp = tmp.rank(axis=1, ascending=False)
    tmp.columns = [t.split("m_")[-1] for t in tmp.columns]
    tmp.columns = [group_dict[c] for c in tmp.columns]
    means = tmp.mean()
    tmp.columns = [t + " $(%.2f)$" % m for t, m in zip(tmp.columns, means.values)]

    fig, ax = plt.subplots(figsize=(width, height), dpi=1000)
    sns.heatmap(tmp.T, cmap="Blues_r", annot=tmp.T.to_numpy(), ax=ax)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig("../08_figures/shaps_over_time.pdf")


# %% Top features for N-En (SHAP):
def append_group_name(feature, feature_class):
    gp = group_abbreviations(feature_class[feature_class.Feature == feature].copy().Group.iloc[0])
    return "{} ({})".format(feature, gp)


def group_abbreviations(grp):
    if grp == "Accruals":
        idx = 3
    elif grp == "Profitability":
        idx = 4
    elif grp == "Quality":
        idx = 1
    elif grp == "Investment":
        idx = 3
    elif grp == "Illiquidity":
        idx = 3
    elif grp == "Informed Trading":
        idx = 4
    elif grp == "Value":
        idx = 3
    elif grp == "Contract":
        idx = 1
    elif grp == "Past Prices":
        idx = 4
    elif grp == "Frictions":
        idx == 4
    elif grp == "Industry":
        idx == 3
    elif grp == "Risk":
        idx = 4
    return grp[:idx]


feature_class = pd.read_parquet("../03_data/features_overview.pq")


to_plot = []
num_feat = 10
for model in ["N-En"]:
    tmp = prediction_dict[model]["SHAPs"]["full"].copy()
    N = tmp.N
    tmp = tmp[[c for c in tmp.columns if not c.startswith("m_")]]
    tmp = tmp[
        [
            c
            for c in tmp.columns
            if c
            not in [
                "m_information-options",
                "m_information-underlying",
                "m_instrument-bucket;instrument-contract",
                "N",
            ]
        ]
    ]
    tmp = tmp.multiply(N, axis=0)  # scale to sum of SHAPs
    means = tmp.sum()
    means = means.divide(means.sum())
    means = means.sort_values(ascending=False).head(num_feat)
    tmp = tmp.divide(tmp.sum(axis=1), axis=0)  # relative
    # means = tmp.mean().sort_values(ascending=False).head(num_feat)
    tmp = tmp[means.index]
    # tmp.columns = [t + " $(%.2f)$" % m for t, m in zip(tmp.columns, means.values)]
    means.index = tmp.columns
    tmp = tmp.stack().to_frame().reset_index()
    tmp.columns = ["date", "Feature", "SHAP"]
    tmp = tmp.set_index("Feature")
    tmp = tmp.loc[means.index]
    tmp = tmp.reset_index()

    tmp["Feature"] = tmp["Feature"].apply(lambda x: append_group_name(x, feature_class))

    fig, ax = plt.subplots(figsize=(width, height), dpi=1000)
    sns.stripplot(
        x="SHAP",
        y="Feature",
        data=tmp,
        ax=ax,
        palette=sns.color_palette("hls", num_feat),
        alpha=1,
    )
    sns.barplot(
        x="SHAP",
        y="Feature",
        data=tmp,
        ax=ax,
        palette=sns.color_palette("hls", num_feat),
        alpha=0.5,
        ci=None,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig("../08_figures/shaps_single_feature.pdf")

# %% Top features for N-En (SHAP) per bucket:
for model in ["N-En"]:

    full = prediction_dict[model]["SHAPs"]["full"].copy()

    N = full.N
    full = full[[c for c in full.columns if not c.startswith("m_")]]
    full = full[
        [
            c
            for c in full.columns
            if c
            not in [
                "m_information-options",
                "m_information-underlying",
                "m_instrument-bucket;instrument-contract",
                "N",
            ]
        ]
    ]
    full = full.multiply(N, axis=0)  # scale to sum of SHAPs
    means = full.sum()
    means = means.divide(means.sum())
    means = means.sort_values(ascending=False)
    means.name = "rank"
    means = means.rank(ascending=False)

    full_sngl_ftr = pd.Series(
        means.reset_index().apply(lambda x: x["index"] + " (" + str(int(x["rank"])) + ")", axis=1).values,
        index=means.index,
    )
    full_sngl_ftr = full_sngl_ftr.to_dict()

    tmp = prediction_dict[model]["SHAPs"]["bucket"].copy()
    tmp.reset_index("bucket", inplace=True)
    to_plot = []
    for idx, ttemp in tmp.groupby("bucket"):
        ttemp.drop(columns="bucket", inplace=True)
        N = ttemp.N
        ttemp = ttemp[[c for c in ttemp.columns if not c.startswith("m_")]]
        ttemp = ttemp[
            [
                c
                for c in ttemp.columns
                if c
                not in [
                    "m_information-options",
                    "m_information-underlying",
                    "m_instrument-bucket;instrument-contract",
                    "N",
                ]
            ]
        ]
        ttemp = ttemp.multiply(N, axis=0)  # scale to sum of SHAPs
        means = ttemp.sum()
        means = means.divide(means.sum())
        means = means.sort_values(ascending=False).head(num_feat)
        ttemp = ttemp.divide(ttemp.sum(axis=1), axis=0)  # relative
        # means = ttemp.mean().sort_values(ascending=False).head(num_feat)
        ttemp = ttemp[means.index]
        # ttemp.columns = [t + " $(%.2f)$" % m for t, m in zip(ttemp.columns, means.values)]
        means.index = ttemp.columns
        ttemp = ttemp.stack().to_frame().reset_index()
        ttemp.columns = ["date", "Feature", "SHAP"]
        ttemp = ttemp.set_index("Feature")
        ttemp = ttemp.loc[means.index]
        ttemp = ttemp.reset_index()
        ttemp["bucket"] = idx
        to_plot.append(ttemp)

    to_plot = pd.concat(to_plot)
    to_plot.set_index("bucket", inplace=True)
    to_plot = to_plot[~to_plot.index.str.contains("nan")]
    to_plot.Feature = to_plot.Feature.apply(lambda x: full_sngl_ftr[x])

    second_page = ["long_term_otm_call", "long_term_otm_put", "short_term_otm_call", "short_term_otm_put"]
    to_plot_2 = to_plot[to_plot.index.get_level_values("bucket").isin(second_page)]
    to_plot_1 = to_plot[~to_plot.index.get_level_values("bucket").isin(second_page)]

    do_one_page = True
    if do_one_page:
        fig, ax = plt.subplots(5, 2, figsize=(width, 15 / 2.54), sharex=False)

        for i, bucket in enumerate(to_plot.index.get_level_values("bucket").unique()):
            if "long" in bucket:
                j = 1
            else:
                j = 0
                i -= 5

            tmp = to_plot[to_plot.index.get_level_values("bucket") == bucket].copy()
            tmp.reset_index(drop=True, inplace=True)
            tmp.set_index("date", inplace=True)
            order = tmp.groupby("Feature").mean().sort_values("SHAP", ascending=False).index.values

            sns.stripplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                ax=ax[i, j],
                order=order,
                palette=sns.color_palette("hls", 12),
                alpha=1,
                orient="v",
                zorder=1,
            )

            sns.barplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                ax=ax[i, j],
                order=order,
                palette=sns.color_palette("hls", 12),
                alpha=0.5,
                ci=None,
                orient="v",
                zorder=0,
            )
            ax[i, j].set_ylabel("")
            ax[i, j].set_xlabel("")
            ax[i, j].tick_params(axis="x", rotation=90, direction="in", pad=-5, labelbottom=True, zorder=2, labelsize=6)
            ax[i, j].set_title(rename_dict[bucket], fontsize=8)
            plt.setp(ax[i, j].get_xticklabels(), va="bottom")

        # for label in ax.get_xticklabels():
        # label.set_va("bottom")

        fig.tight_layout()
        fig.savefig("../08_figures/shaps_single_buckets.pdf")

    else:
        fig, ax = plt.subplots(3, 2, figsize=(width, 15 / 2.54), sharex=False)

        for i, bucket in enumerate(to_plot_1.index.get_level_values("bucket").unique()):
            if "long" in bucket:
                j = 1
            else:
                j = 0
                i -= 3

            tmp = to_plot_1[to_plot_1.index.get_level_values("bucket") == bucket].copy()
            tmp.reset_index(drop=True, inplace=True)
            tmp.set_index("date", inplace=True)
            order = tmp.groupby("Feature").mean().sort_values("SHAP", ascending=False).index.values

            sns.stripplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                ax=ax[i, j],
                order=order,
                palette=sns.color_palette("hls", 12),
                alpha=1,
                orient="v",
                zorder=1,
            )

            sns.barplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                ax=ax[i, j],
                order=order,
                palette=sns.color_palette("hls", 12),
                alpha=0.5,
                ci=None,
                orient="v",
                zorder=0,
            )
            ax[i, j].set_ylabel("")
            ax[i, j].set_xlabel("")
            ax[i, j].tick_params(axis="x", rotation=90, direction="in", pad=-10, labelbottom=True, zorder=2)
            ax[i, j].set_title(rename_dict[bucket], fontsize=8)
            plt.setp(ax[i, j].get_xticklabels(), va="bottom")

        # for label in ax.get_xticklabels():
        # label.set_va("bottom")

        fig.tight_layout()
        fig.savefig("../08_figures/shaps_single_buckets_1.pdf")

        fig, ax = plt.subplots(2, 2, figsize=(width, 10 / 2.54), sharex=False)
        for i, bucket in enumerate(to_plot_2.index.get_level_values("bucket").unique()):

            if "long" in bucket:
                j = 1
            else:
                j = 0
                i -= 2

            tmp = to_plot_2[to_plot_2.index.get_level_values("bucket") == bucket].copy()
            tmp.reset_index(drop=True, inplace=True)
            tmp.set_index("date", inplace=True)
            order = tmp.groupby("Feature").mean().sort_values("SHAP", ascending=False).index.values

            sns.stripplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                ax=ax[i, j],
                order=order,
                palette=sns.color_palette("hls", 12),
                alpha=1,
                orient="v",
                zorder=1,
            )
            sns.barplot(
                x="Feature",
                y="SHAP",
                data=tmp,
                ax=ax[i, j],
                order=order,
                palette=sns.color_palette("hls", 12),
                alpha=0.5,
                ci=None,
                orient="v",
                zorder=0,
            )
            ax[i, j].set_ylabel("")
            ax[i, j].set_xlabel("")
            ax[i, j].tick_params(axis="x", rotation=90, direction="in", pad=-10, labelbottom=True, zorder=2)
            ax[i, j].set_title(rename_dict[bucket], fontsize=8)
            plt.setp(ax[i, j].get_xticklabels(), va="bottom")

        fig.tight_layout()
        fig.savefig("../08_figures/shaps_single_buckets_2.pdf")


# %%
# Bucket performance
for i, model in enumerate(["L-En", "N-En"]):
    tmp = prediction_dict[model]["predictions"]
    tmp = tmp.reset_index()
    tmp = tmp.drop(columns=["permno", "port"])
    tmp = tmp.rename(columns={"predicted": model})
    tmp = tmp.set_index(["date", "optionid"])
    if i == 0:
        to_plot = tmp.copy()
    else:
        to_plot = to_plot.merge(tmp.drop(columns=["target"]), on=["date", "optionid"])

to_plot = to_plot.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])
to_plot = to_plot.set_index(["date", "optionid"])


# DM test for differences between the two:
dm_test = to_plot.groupby("bucket").apply(lambda x: DieboldMariano(x, 12))
stars = []
for c in dm_test.squeeze():
    if c > 3.09:
        stars.append("***")
    elif c > 2.326:
        stars.append("**")
    elif c > 1.645:
        stars.append("*")
    else:
        stars.append("")
stars = pd.Series(data=stars, index=dm_test.index)


tables = []
for model in ["L-En", "N-En"]:
    tmp = to_plot.groupby("bucket").apply(lambda x: TS_R2(x[["target", model]]))
    tmp.columns = ["score"]
    tmp = tmp.reset_index()
    if model == "N-En":
        n_en_bucket = to_plot.copy()
    tmp = tmp.rename(columns={"predicted": "score"})
    tmp = tmp[~tmp.bucket.str.contains("nan")]
    # rearrange to put in table:
    ttm = []
    moneyness = []
    types = []
    for i, bucket in enumerate(tmp.bucket.str.split("_").values):
        if "long" in bucket:
            ttm.append("Long-term")
        else:
            ttm.append("Short-term")
        if "itm" in bucket:
            moneyness.append("ITM")
        elif "atm" in bucket:
            moneyness.append("ATM")
        elif "otm" in bucket:
            moneyness.append("OTM")
        if "call" in bucket:
            types.append("Call")
        elif "put" in bucket:
            types.append("Put")
        else:
            types.append("")
    tmp["TTM"] = ttm
    tmp["Type"] = [m + " " + t for m, t in zip(moneyness, types)]
    tmp = tmp.set_index(["bucket", "TTM", "Type"])
    tmp = tmp["score"]
    tmp = tmp.reset_index().set_index("bucket")
    if model == "N-En":
        n_en_bucket = tmp.copy()
        n_en_bucket["model"] = "contract"
    tmp["model"] = model
    tables.append(tmp.copy())
tables = pd.concat(tables, axis=0)

fig, ax = plt.subplots(2, 1, figsize=(width, height), sharex=True, sharey=True)
for i, ttm in enumerate(["Short-term", "Long-term"]):
    g = sns.barplot(x="Type", y="score", hue="model", data=tables[tables.TTM == ttm], palette=sns_palette(2), ax=ax[i])
    ax[i].set_xlabel("")
    ax[i].set_ylabel(ttm)
    ax[i].axhline(0, color="k", ls="--", lw=1, label="_nolegend_")
    if i == 0:
        ax[i].legend([], frameon=False)
    else:
        g.legend_.set_title(None)
        ax[i].legend(frameon=False, loc="lower right", ncol=2)
    ax[i].grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax[i].set_ylim([-0.03, 0.05])
    ax[i].set_axisbelow(True)
    for j, (idx, val) in enumerate(tables[tables.TTM == ttm].score.iteritems()):
        ax[i].annotate(stars.loc[idx].values[0], (j, 0.041), ha="center", color="black")
fig.subplots_adjust(hspace=0.05)
fig.savefig("../08_figures/r2_bucket.pdf", bbox_inches="tight")


# %%
# Bucket performance (XS)
for i, model in enumerate(["L-En", "N-En"]):
    tmp = prediction_dict[model]["predictions"]
    tmp = tmp.reset_index()
    tmp = tmp.drop(columns=["permno", "port"])
    tmp = tmp.rename(columns={"predicted": model})
    tmp = tmp.set_index(["date", "optionid"])
    if i == 0:
        to_plot = tmp.copy()
    else:
        to_plot = to_plot.merge(tmp.drop(columns=["target"]), on=["date", "optionid"])

to_plot = to_plot.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])
to_plot = to_plot.set_index(["date", "optionid"])


# DM test for differences between the two:
dm_test = to_plot.groupby("bucket").apply(lambda x: DieboldMariano_XS(x, 12))
stars = []
for c in dm_test.squeeze():
    if c > 3.09:
        stars.append("***")
    elif c > 2.326:
        stars.append("**")
    elif c > 1.645:
        stars.append("*")
    else:
        stars.append("")
stars = pd.Series(data=stars, index=dm_test.index)


tables = []
for model in ["L-En", "N-En"]:
    tmp = to_plot.groupby("bucket").apply(lambda x: XS_R2(x[["target", model]]))
    tmp.columns = ["score"]
    tmp = tmp.reset_index()
    tmp = tmp.rename(columns={"predicted": "score"})
    tmp = tmp[~tmp.bucket.str.contains("nan")]
    # rearrange to put in table:
    ttm = []
    moneyness = []
    types = []
    for i, bucket in enumerate(tmp.bucket.str.split("_").values):
        if "long" in bucket:
            ttm.append("Long-term")
        else:
            ttm.append("Short-term")
        if "itm" in bucket:
            moneyness.append("ITM")
        elif "atm" in bucket:
            moneyness.append("ATM")
        elif "otm" in bucket:
            moneyness.append("OTM")
        if "call" in bucket:
            types.append("Call")
        elif "put" in bucket:
            types.append("Put")
        else:
            types.append("")
    tmp["TTM"] = ttm
    tmp["Type"] = [m + " " + t for m, t in zip(moneyness, types)]
    tmp = tmp.set_index(["bucket", "TTM", "Type"])
    tmp = tmp["score"]
    tmp = tmp.reset_index().set_index("bucket")
    if model == "N-En":
        n_en_bucket_xs = tmp.copy()
        n_en_bucket_xs["model"] = "contract"
    tmp["model"] = model
    tables.append(tmp.copy())
tables = pd.concat(tables, axis=0)

fig, ax = plt.subplots(2, 1, figsize=(width, height), sharex=True, sharey=True)
for i, ttm in enumerate(["Short-term", "Long-term"]):
    g = sns.barplot(x="Type", y="score", hue="model", data=tables[tables.TTM == ttm], palette=sns_palette(2), ax=ax[i])
    ax[i].set_xlabel("")
    ax[i].set_ylabel(ttm)
    ax[i].axhline(0, color="k", ls="--", lw=1, label="_nolegend_")
    if i == 0:
        ax[i].legend([], frameon=False)
    else:
        g.legend_.set_title(None)
        ax[i].legend(frameon=False, loc="lower right", ncol=2)
    ax[i].grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax[i].set_ylim([-0.03, 0.05])
    ax[i].set_axisbelow(True)
    for j, (idx, val) in enumerate(tables[tables.TTM == ttm].score.iteritems()):
        ax[i].annotate(stars.loc[idx].values[0], (j, 0.043), ha="center", color="black")
fig.subplots_adjust(hspace=0.05)
fig.savefig("../08_figures/r2_xs_bucket.pdf", bbox_inches="tight")


# %%
# Bucket PORTFOLIO performance
tables = []
for model in ["N-En"]:
    tmp = prediction_dict[model]["predictions"]
    tmp = tmp.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])

    # ---- value-weighting
    tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi"]], on=["date", "optionid"])
    weighted = tmp[["predicted", "target"]].multiply(tmp["doi"], axis=0)
    tmp[weighted.columns] = weighted
    tmp = tmp.drop(columns=["optionid"])

    # weights per permno
    total_weights = tmp.groupby(["date", "permno", "bucket"]).doi.sum()  # denominator
    total_weights.name = "summed_weights"
    tmp = tmp.groupby(["date", "permno", "bucket"]).sum()  # numerator
    tmp = tmp.divide(total_weights, axis=0)
    tmp = tmp.drop(columns=["doi"])

    tmp = tmp.groupby("bucket").apply(lambda x: TS_R2(x[["target", "predicted"]]))
    tmp = tmp.reset_index()
    tmp = tmp.rename(columns={"predicted": "score"})
    tmp = tmp[~tmp.bucket.str.contains("nan")]

    # rearrange to put in table:
    ttm = []
    moneyness = []
    types = []
    for i, bucket in enumerate(tmp.bucket.str.split("_").values):
        if "long" in bucket:
            ttm.append("Long-term")
        else:
            ttm.append("Short-term")
        if "itm" in bucket:
            moneyness.append("ITM")
        elif "atm" in bucket:
            moneyness.append("ATM")
        elif "otm" in bucket:
            moneyness.append("OTM")
        if "call" in bucket:
            types.append("Call")
        elif "put" in bucket:
            types.append("Put")
        else:
            types.append("")
    tmp["TTM"] = ttm
    tmp["Type"] = [m + " " + t for m, t in zip(moneyness, types)]
    tmp = tmp.set_index(["bucket", "TTM", "Type"])
    tmp = tmp["score"]
    tmp = tmp.reset_index().set_index("bucket")
    tmp["model"] = "portfolio"
    tmp = pd.concat((n_en_bucket, tmp), axis=0)
    tables.append(tmp.copy())

tables = pd.concat(tables, axis=0)

fig, ax = plt.subplots(2, 1, figsize=(width, height), sharex=True, sharey=True)
for i, ttm in enumerate(["Short-term", "Long-term"]):
    g = sns.barplot(x="Type", y="score", hue="model", data=tables[tables.TTM == ttm], palette=sns_palette(2), ax=ax[i])
    ax[i].set_xlabel("")
    ax[i].set_ylabel(ttm)
    ax[i].axhline(0, color="k", ls="--", lw=1, label="_nolegend_")
    if i == 0:
        g.legend_.set_title(None)
        ax[i].legend(frameon=False)
    else:
        ax[i].legend([], frameon=False)
    ax[i].grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax[i].set_axisbelow(True)
fig.subplots_adjust(hspace=0.05)
fig.savefig("../08_figures/r2_portfolios.pdf", bbox_inches="tight")


# %%
# Bucket PORTFOLIO performance --- XS
tables = []
for model in ["N-En"]:
    tmp = prediction_dict[model]["predictions"]
    tmp = tmp.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])

    # ---- value-weighting
    tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi"]], on=["date", "optionid"])
    weighted = tmp[["predicted", "target"]].multiply(tmp["doi"], axis=0)
    tmp[weighted.columns] = weighted
    tmp = tmp.drop(columns=["optionid"])

    # weights per permno
    total_weights = tmp.groupby(["date", "permno", "bucket"]).doi.sum()  # denominator
    total_weights.name = "summed_weights"
    tmp = tmp.groupby(["date", "permno", "bucket"]).sum()  # numerator
    tmp = tmp.divide(total_weights, axis=0)
    tmp = tmp.drop(columns=["doi"])

    tmp = tmp.groupby("bucket").apply(lambda x: XS_R2(x[["target", "predicted"]]))
    tmp = tmp.reset_index()
    tmp = tmp.rename(columns={"predicted": "score"})
    tmp = tmp[~tmp.bucket.str.contains("nan")]

    # rearrange to put in table:
    ttm = []
    moneyness = []
    types = []
    for i, bucket in enumerate(tmp.bucket.str.split("_").values):
        if "long" in bucket:
            ttm.append("Long-term")
        else:
            ttm.append("Short-term")
        if "itm" in bucket:
            moneyness.append("ITM")
        elif "atm" in bucket:
            moneyness.append("ATM")
        elif "otm" in bucket:
            moneyness.append("OTM")
        if "call" in bucket:
            types.append("Call")
        elif "put" in bucket:
            types.append("Put")
        else:
            types.append("")
    tmp["TTM"] = ttm
    tmp["Type"] = [m + " " + t for m, t in zip(moneyness, types)]
    tmp = tmp.set_index(["bucket", "TTM", "Type"])
    tmp = tmp["score"]
    tmp = tmp.reset_index().set_index("bucket")
    tmp["model"] = "portfolios"
    tmp = pd.concat((n_en_bucket_xs, tmp), axis=0)
    tables.append(tmp.copy())

tables = pd.concat(tables, axis=0)

fig, ax = plt.subplots(2, 1, figsize=(width, height), sharex=True, sharey=True)
for i, ttm in enumerate(["Short-term", "Long-term"]):
    g = sns.barplot(x="Type", y="score", hue="model", data=tables[tables.TTM == ttm], palette=sns_palette(2), ax=ax[i])
    ax[i].set_xlabel("")
    ax[i].set_ylabel(ttm)
    ax[i].axhline(0, color="k", ls="--", lw=1, label="_nolegend_")
    if i == 0:
        g.legend_.set_title(None)
        ax[i].legend(frameon=False)
    else:
        ax[i].legend([], frameon=False)
    ax[i].grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax[i].set_axisbelow(True)
fig.subplots_adjust(hspace=0.05)
fig.savefig("../08_figures/r2_portfolios_xs.pdf", bbox_inches="tight")


# %%
# Check the distribution of the correlation of expected returns across buckets for the same underlyings
tmp = prediction_dict["N-En"]["predictions"]
tmp = tmp.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])

correlations = []
permnos = tmp.permno.unique()
for i, permno in enumerate(permnos):
    if i % 100 == 0:
        print(f"{i}/{len(permnos)}")
    a = tmp[tmp.permno == permno]
    a = a.pivot_table(values="predicted", index="date", columns="bucket", aggfunc="mean")
    a.loc[:, (len(a) - a.isnull().sum()) < 10] = np.nan
    a = a.corr().stack()
    a = a.to_frame(name="corr")
    a.index.names = ["A", "B"]
    a["permno"] = permno
    correlations.append(a)
correlations = pd.concat(correlations)
correlations = correlations[~correlations.index.get_level_values("A").str.contains("nan")]
correlations = correlations[~correlations.index.get_level_values("B").str.contains("nan")]

fig, ax = plt.subplots(figsize=(width, height), dpi=800)
sns.heatmap(correlations.groupby(["A", "B"]).mean()["corr"].unstack(), ax=ax)
ax.set_xlabel("")
ax.set_xticklabels([c.get_text().replace("_", " ") for c in ax.get_xticklabels()])
ax.set_ylabel("")
ax.set_yticklabels([c.get_text().replace("_", " ") for c in ax.get_xticklabels()])
fig.tight_layout()
fig.savefig("../08_figures/expected_return_consistency.pdf")


tmp = correlations.loc["short_term_atm"].loc["long_term_atm"]["corr"].to_frame()
tmp["type"] = "Short vs. Long-term"
tmp2 = correlations.loc["short_term_otm_put"].loc["short_term_itm_call"]["corr"].to_frame()
tmp2["type"] = "OTM Puts vs. ITM Calls"
tmp3 = correlations.loc["short_term_otm_call"].loc["short_term_itm_put"]["corr"].to_frame()
tmp3["type"] = "OTM Calls vs. ITM Puts"
tmp = pd.concat((tmp, tmp2, tmp3), ignore_index=True)

fig, ax = plt.subplots(1, 2, figsize=(width, height * 0.8), dpi=800, sharex=True, sharey=False)
sns.histplot(
    data=tmp[tmp.type == "Short vs. Long-term"], x="corr", stat="probability", ax=ax[0], alpha=0.4, color="k", bins=25
)
sns.histplot(
    data=tmp[tmp.type != "Short vs. Long-term"],
    x="corr",
    hue="type",
    stat="probability",
    ax=ax[1],
    alpha=0.4,
    palette="husl",
    bins=25,
)
ax[0].set_xlabel("Correlation by Underlying")
ax[1].set_xlabel("Correlation by Underlying")
ax[1].set_ylabel("")
ax[0].legend(["Short vs. Long-term ATM"], frameon=False)
ax[1].legend(["OTM Puts vs. ITM Calls", "OTM Calls vs. ITM Puts"], frameon=False)
fig.tight_layout()
fig.savefig("../08_figures/expected_return_consistency_buckets.pdf")
