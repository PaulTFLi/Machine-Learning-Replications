# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
""" Comparative analyses for rolling vs. expanding training windows. """


# %%
# Setup
import os
import pandas as pd
import numpy as np
from scoring import ClarkWest, XS_R2, TS_R2
from joblib import load

from pylatex import Tabular, MultiColumn, Math
from pylatex.utils import NoEscape

from scipy.stats import ttest_ind
import statsmodels.api as sm

import matplotlib.pyplot as plt

# import matplotlib
import seaborn as sns

from analysis_setup import sns_palette, width, height, analysisLoc, modelLoc, prediction_dict

class_groups = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups.pq"), columns=["date", "optionid", "type", "bucket"]
)
class_groups_with_info = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups_and_info.pq"),
    columns=["date", "optionid", "doi"],
)


# %%
# Functions
def ts_latex_table(full_sample, cp_sample, save_name: str, num_format="%.4f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    table = Tabular("".join(["l"] + ["c"] * (full_sample.shape[1] - 1) + ["c"]), booktabs=True)
    table.add_row(
        [
            "",
            MultiColumn(4, align="c", data=full_sample.columns.get_level_values(0)[0]),
            "",
            MultiColumn(4, align="c", data=full_sample.columns.get_level_values(0)[-3]),
            "",
            "",
        ]
    )
    to_add = [""] + ["" if "EMPTY" in c else c for c in full_sample.columns.get_level_values(1).tolist()]
    table.add_row([MultiColumn(1, align="c", data=c) for c in to_add])
    table.add_hline(start=2, end=5)
    table.add_hline(start=7, end=10)
    table.add_hline(start=12, end=12)

    for i, (idx, row) in enumerate(full_sample.iterrows()):
        to_add = []
        for col, num in row.iteritems():
            if isinstance(num, float):
                if np.isnan(num):
                    to_add.append("")
                else:
                    if idx == "tval":
                        num = "%.2f" % num
                        to_add.append(math(r"\footnotesize{(%s)}" % num))
                    else:
                        num = num_format % num
                        to_add.append(math(num))
            else:
                to_add.append(num)
        if idx != "tval":
            if idx == "H-L":
                table.add_hline()
            table.add_row([idx] + to_add)
        else:
            table.add_row([""] + to_add)

    # call/put stuff
    table.add_empty_row()
    for i, (idx, row) in enumerate(cp_sample.iterrows()):
        to_add = []
        for col, num in row.iteritems():
            if isinstance(num, float):
                if np.isnan(num):
                    to_add.append("")
                else:
                    if idx == "tval":
                        num = "%.2f" % num
                        to_add.append(math(r"\footnotesize{(%s)}" % num))
                    else:
                        num = num_format % num
                        to_add.append(math(num))
            else:
                to_add.append(num)
        if idx != "tval":
            if idx == "H-L":
                table.add_hline()
            table.add_row([idx] + to_add)
        else:
            table.add_row([""] + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


def Lo_tval_SR(returns, m=12):
    mean = returns.mean()
    var = returns.var()
    T = len(returns)

    psi = [returns - mean, (returns - mean) ** 2 - var]
    psi = pd.concat(psi, axis=1)
    psi.columns = ["psi1", "psi2"]
    psi = psi.to_numpy()

    # weights
    weights = []
    for j in np.arange(0, m + 1):
        weights.append(1 - j / (m + 1))

    # omegas
    omegas = []
    for j in np.arange(0, m + 1):
        running_sum = np.zeros((2, 2))
        for t in np.arange(j + 1, T):
            running_sum += psi[t] * np.atleast_2d(psi[t - j]).T
        running_sum /= len(returns)
        omegas.append(running_sum)

    # sigma
    sigma = np.array(np.zeros((2, 2)))
    for j in np.arange(0, m + 1):
        sigma += weights[j] * (omegas[j] + omegas[j].T)

    # derivatives
    derivs = np.array([1 / np.sqrt(var), -(mean) / (2 * np.sqrt(var) ** 3)])
    derivs = np.atleast_2d(derivs)

    # variance + tstat
    estimated_variance = derivs.dot(sigma).dot(derivs.T)[0][0]
    bse = np.sqrt(estimated_variance / T)
    tval = (mean / np.sqrt(var)) / bse

    return tval


# %%
# ROLLING
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
add_on = "rolling"
rolling_dict = load(os.path.join(modelLoc, f"prediction_dict_{add_on}.pkl"))["N-En"]
prediction_dict = prediction_dict["N-En"]


# %% OOS R2 full sample
for i, dict in enumerate([prediction_dict, rolling_dict]):
    target = dict["predictions"][["optionid", "target"]].copy()
    tmp = dict["predictions"]["predicted"].copy()
    if i == 0:
        tmp.name = "Expanding"
    else:
        tmp.name = "Rolling"
    tmp = pd.concat((tmp, target), axis=1)
    if i == 0:
        to_plot = tmp
    else:
        to_plot = to_plot.merge(tmp.drop(columns=["target"]), on=["date", "optionid"])

# add type (call/put) information:
to_plot = to_plot.reset_index()
to_plot = to_plot.merge(class_groups[["date", "optionid", "type"]], on=["date", "optionid"])
to_plot = to_plot.drop(columns=["optionid"])
to_plot = to_plot.set_index("date")


# XS R2
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

cw_test = to_plot.groupby("type").apply(lambda x: ClarkWest(x, 12, benchmark_type="xs", cw_adjust=False))
cw_test.loc["all"] = ClarkWest(to_plot.drop(columns=["type"]), 12, benchmark_type="xs", cw_adjust=False)
cw_test = cw_test.stack()
xs_stars = []
for c in cw_test:
    if c > 3.09:
        xs_stars.append("***")
    elif c > 2.326:
        xs_stars.append("**")
    elif c > 1.645:
        xs_stars.append("*")
    else:
        xs_stars.append("")
xs_stars = pd.Series(data=xs_stars, index=cw_test.index)


# TS R2
calls = TS_R2(to_plot[to_plot["type"] == "call"].drop(columns=["type"]))
calls = calls.to_frame()
calls["type"] = "Call"
puts = TS_R2(to_plot[to_plot["type"] == "put"].drop(columns=["type"]))
puts = puts.to_frame()
puts["type"] = "Put"
r2 = TS_R2(to_plot.drop(columns=["type"]))
r2 = r2.to_frame()
r2["type"] = "All"
r2 = pd.concat((calls, puts, r2), axis=0)
r2 = r2.reset_index()
r2.columns = ["model", "score", "type"]
# r2["type"] = r"$R^2_{OS;XS}$"

cw_test = to_plot.groupby("type").apply(lambda x: ClarkWest(x, 12, benchmark_type="zero", cw_adjust=False))
cw_test.loc["all"] = ClarkWest(to_plot.drop(columns=["type"]), 12, benchmark_type="zero", cw_adjust=False)
cw_test = cw_test.stack()
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


fig, axes = plt.subplots(1, 2, figsize=(width, height * 0.8), dpi=1000, sharey=True)
sns.barplot(x="model", y="score", hue="type", data=r2, ax=axes[0], palette=sns_palette(3))
sns.barplot(x="model", y="score", hue="type", data=xs_r2, ax=axes[1], palette=sns_palette(3))
for ax in axes:
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax.set_axisbelow(True)
    for i, (_, txt) in enumerate(stars.iteritems()):
        ax.annotate(txt, (i, 0.0), ha="center", color="white")
axes[1].legend([], frameon=False, title=None)
axes[0].legend(frameon=False, title=None)
axes[0].set_title("$R^2_{OS}$")
axes[1].set_title("$R^2_{OS;XS}$")
fig.tight_layout()
fig.savefig(f"../08_figures/r2_{add_on}.pdf", bbox_inches="tight")


# %%
# Trading strategy using **value-weights*:
to_plot = []
model_returns = {}
prediction_dict = {"Expanding": prediction_dict, "Rolling": rolling_dict}
min_date = rolling_dict["predictions"].index.min()
for model in ["Expanding", "Rolling"]:
    tmp = prediction_dict[model]["predictions"].copy()
    tmp = tmp.loc[min_date:]

    # ---- DOI-weighted portfolios, across underlyings
    tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi"]], on=["date", "optionid"])
    weighted = tmp[["predicted", "target"]].multiply(tmp["doi"], axis=0)
    tmp[weighted.columns] = weighted
    tmp = tmp.drop(columns=["optionid"])

    total_weights = tmp.groupby(["date", "port"]).doi.sum()  # denominator
    total_weights.name = "summed_weights"
    tmp = tmp.groupby(["date", "port"]).sum()  # numerator
    tmp = tmp.divide(total_weights, axis=0)
    tmp = tmp.drop(columns=["doi"])

    output = []
    for col in ["predicted", "target"]:
        returns = tmp[col].unstack()
        returns.columns = "Lo 2 3 4 5 6 7 8 9 Hi".split()
        returns["H-L"] = returns["Hi"] - returns["Lo"]
        returns = returns.stack()
        returns.name = col
        returns.index.names = ["date", "port"]
        output.append(returns)
    output = pd.concat(output, axis=1)
    model_returns[model] = output["target"].copy()

    grouper = output.groupby("port")
    strat = grouper.mean()
    strat.columns = ["Pred", "Avg"]
    strat["SD"] = grouper.target.std().tolist()
    strat *= 100
    strat["SR"] = strat["Avg"] / strat["SD"]
    strat.columns = pd.MultiIndex.from_arrays([[model] * strat.shape[1], strat.columns])
    tvals = [
        np.nan,
        grouper.apply(
            lambda x: sm.OLS(x["target"], np.ones(x["target"].shape))
            .fit(cov_type="HAC", cov_kwds={"maxlags": 12})
            .tvalues
        ).loc["H-L", "const"],
        np.nan,
        Lo_tval_SR(output[output.index.get_level_values("port") == "H-L"]["target"]),
    ]
    strat.loc["tval"] = tvals
    to_plot.append(strat)

model_returns = pd.concat(model_returns, axis=1)

to_plot = pd.concat(to_plot, axis=1)
to_plot.insert(4, "EMPTY_0", "")
to_plot["EMPTY_1"] = ""
val = model_returns.groupby("port").apply(lambda x: ttest_ind(x["Expanding"], x["Expanding"])[0])
val["tval"] = 0
to_plot[("", "Exp. vs. Roll")] = ""
to_plot.loc[val.abs() > 1.65, ("", "Exp. vs. Roll")] = "*"
to_plot.loc[val.abs() > 1.96, ("", "Exp. vs. Roll")] = "**"
to_plot.loc[val.abs() > 2.56, ("", "Exp. vs. Roll")] = "***"

full_sample = to_plot.copy()


# ---- split by puts and calls (only hml)
to_plot = []
model_returns = {}
col = "type"
for model in ["Expanding", "Rolling"]:
    tmp = prediction_dict[model]["predictions"].copy()
    tmp = tmp.loc[min_date:]
    tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])

    tmp["port"] = tmp.groupby(["date", col]).predicted.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    tmp = tmp.set_index("date")

    # ---- DOI-weighted portfolios, across underlyings
    tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi"]], on=["date", "optionid"])
    # tmp = tmp[tmp.bucket == "short_term_otm_call"]
    # tmp = tmp.drop(columns=["bucket"])
    weighted = tmp[["predicted", "target"]].multiply(tmp["doi"], axis=0)
    tmp[weighted.columns] = weighted
    tmp = tmp.drop(columns=["optionid"])

    total_weights = tmp.groupby(["date", col, "port"]).doi.sum()  # denominator
    total_weights.name = "summed_weights"
    tmp = tmp.groupby(["date", col, "port"]).sum()  # numerator
    tmp = tmp.divide(total_weights, axis=0)
    tmp = tmp.drop(columns=["doi"])

    output = []
    for target_col in ["predicted", "target"]:
        # returns = tmp.groupby(["date", col, "port"])[target_col].mean()
        returns = tmp[target_col].copy()
        returns = returns.unstack()
        returns.columns = "Lo 2 3 4 5 6 7 8 9 Hi".split()
        returns["H-L"] = returns["Hi"] - returns["Lo"]
        returns = returns.stack()
        returns.name = target_col
        returns.index.names = ["date", col, "port"]
        output.append(returns)
    output = pd.concat(output, axis=1)
    output = output[output.index.get_level_values("port") == "H-L"]
    model_returns[model] = output["target"].copy()

    grouper = output.groupby([col, "port"])
    strat = grouper.mean()
    strat.columns = ["Pred", "Avg"]
    strat["SD"] = grouper.target.std().tolist()
    strat *= 100
    strat["SR"] = strat["Avg"] / strat["SD"]
    strat.columns = pd.MultiIndex.from_arrays([[model] * strat.shape[1], strat.columns])
    to_plot.append(strat)

model_returns = pd.concat(model_returns, axis=1)

to_plot = pd.concat(to_plot, axis=1)

val = model_returns.groupby([col, "port"]).apply(lambda x: ttest_ind(x["Expanding"], x["Rolling"])[0])
to_plot.insert(4, "EMPTY_0", "")
to_plot["EMPTY_1"] = ""
to_plot[("", "Exp. vs. Roll")] = ""
to_plot.loc[val.abs() > 1.65, ("", "Exp. vs. Roll")] = "*"
to_plot.loc[val.abs() > 1.96, ("", "Exp. vs. Roll")] = "**"
to_plot.loc[val.abs() > 2.56, ("", "Exp. vs. Roll")] = "***"
to_plot = to_plot.reset_index()

cp_sample = to_plot.copy()
cp_sample = cp_sample.drop(columns=["port"])
cp_sample = cp_sample.set_index("type")


ts_latex_table(full_sample, cp_sample, f"trading_strat_vw_{add_on}", num_format="%.3f")


# %%
