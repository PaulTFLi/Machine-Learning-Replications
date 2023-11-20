# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
# Setup
import os
import pandas as pd
import numpy as np
from scoring import ClarkWest, XS_R2
from joblib import load

from pylatex import Tabular, MultiColumn, Math, MultiRow
from pylatex.utils import NoEscape

from scipy.stats import ttest_ind
import statsmodels.api as sm

import matplotlib.pyplot as plt

# import matplotlib
import seaborn as sns

from analysis_setup import sns_palette, width, height, analysisLoc, modelLoc

class_groups = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups_weekly.pq"), columns=["date", "optionid", "type"]
)
class_groups_with_info = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups_and_info_weekly.pq"),
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
# WEEKLY RETURNS
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
add_on = "weekly"
prediction_dict = load(os.path.join(modelLoc, f"prediction_dict_{add_on}.pkl"))


# %% OOS R2 full sample
to_plot = []
for i, model in enumerate(["L-En", "N-En"]):
    print(i)
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
for model in ["L-En", "N-En"]:
    tmp = prediction_dict[model]["class_scores"]["type"].copy()
    tmp.loc["all"] = prediction_dict[model]["scores"].values
    tmp = tmp.reset_index()
    tmp["model"] = model.split("_")[0]
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot)
to_plot["type"] = to_plot["type"].replace({"all": "All", "call": "Call", "put": "Put"})

r2 = to_plot.copy()


# OOS R2 full sample ---- CROSS-SECTIONAL VERSION FOLLOWING
# Rapach et al. "Firm Characteristics and Expected Returns", 2021
# Addresses Weak vs. Strong factors to soME Degree.
to_plot = []
for i, model in enumerate(["L-En", "N-En"]):
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
fig.tight_layout()
fig.savefig(f"../08_figures/r2_{add_on}.pdf", bbox_inches="tight")


# %%
# Trading strategy using **value-weights*:
to_plot = []
model_returns = {}
for model in ["L-En", "N-En"]:
    tmp = prediction_dict[model]["predictions"].copy()

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
val = model_returns.groupby("port").apply(lambda x: ttest_ind(x["N-En"], x["L-En"])[0])
val["tval"] = 0
to_plot[("", "N vs. L")] = ""
to_plot.loc[val.abs() > 1.65, ("", "N vs. L")] = "*"
to_plot.loc[val.abs() > 1.96, ("", "N vs. L")] = "**"
to_plot.loc[val.abs() > 2.56, ("", "N vs. L")] = "***"

full_sample = to_plot.copy()


# ---- split by puts and calls (only hml)
to_plot = []
model_returns = {}
col = "type"
for model in ["L-En", "N-En"]:
    tmp = prediction_dict[model]["predictions"].copy()
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

val = model_returns.groupby([col, "port"]).apply(lambda x: ttest_ind(x["N-En"], x["L-En"])[0])
to_plot.insert(4, "EMPTY_0", "")
to_plot["EMPTY_1"] = ""
to_plot[("", "N vs. L")] = ""
to_plot.loc[val.abs() > 1.65, ("", "N vs. L")] = "*"
to_plot.loc[val.abs() > 1.96, ("", "N vs. L")] = "**"
to_plot.loc[val.abs() > 2.56, ("", "N vs. L")] = "***"
to_plot = to_plot.reset_index()

cp_sample = to_plot.copy()
cp_sample = cp_sample.drop(columns=["port"])
cp_sample = cp_sample.set_index("type")


ts_latex_table(full_sample, cp_sample, f"trading_strat_vw_{add_on}", num_format="%.3f")


# %% Trading strategy for the earnings vs non-earnings events
def full_ts(flag_vw=False, idx=None, additional=[], renaming="Full"):
    to_plot = []
    model_returns = {}
    for model in ["L-En", "N-En"]:
        tmp = prediction_dict[model]["predictions"].copy()
        tmp = tmp.reset_index()
        tmp.date = tmp.date.dt.week + tmp.date.dt.year * 100
        if not (idx is None):
            name_idx = list(set(additional.columns) - set(["date", "permno"]))[0]

            tmp = tmp.merge(additional, left_on=["date", "permno"], right_on=["date", "permno"], how="left")
            tmp[name_idx] = tmp[name_idx].fillna(0)
            tmp = tmp.loc[tmp[name_idx] == idx, :]
            tmp = tmp.drop(columns=[name_idx]).set_index("date")
        if flag_vw:
            cg = class_groups_with_info[["date", "optionid", "doi"]]
            cg.date = cg.date.dt.week + cg.date.dt.year * 100
            tmp = tmp.merge(cg, on=["date", "optionid"])
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
            returns = tmp.groupby(["date", "port"])[col].mean()
            returns = returns.unstack()
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
        strat.loc["t-val"] = tvals
        to_plot.append(strat)

    model_returns = pd.concat(model_returns, axis=1)

    to_plot = pd.concat(to_plot, axis=1)
    to_plot.insert(4, "EMPTY_0", "")
    to_plot["EMPTY_1"] = ""
    val = model_returns.groupby("port").apply(lambda x: ttest_ind(x["N-En"], x["L-En"])[0])
    val["t-val"] = 0
    to_plot[("", "N vs. L")] = ""
    to_plot.loc[val.abs() > 1.65, ("", "N vs. L")] = "*"
    to_plot.loc[val.abs() > 1.96, ("", "N vs. L")] = "**"
    to_plot.loc[val.abs() > 2.56, ("", "N vs. L")] = "***"

    to_plot = to_plot.loc[["H-L", "t-val"]]
    to_plot = to_plot.rename(index={"H-L": renaming})
    to_plot = to_plot.reset_index()
    to_plot = to_plot.rename(columns={"port": "Sample"})
    if flag_vw:
        to_plot["Weighting"] = "VW"
    else:
        to_plot["Weighting"] = "EW"
    to_plot = to_plot.set_index(["Weighting", "Sample"])

    return to_plot


earnings_news = pd.read_parquet("../03_data/firm_level_events.pq", columns=["date", "permno", "earnings", "news"])
earnings = earnings_news[["date", "permno", "earnings"]].dropna()
news = earnings_news[["date", "permno", "news"]].dropna()

earnings["date"] = earnings["date"] - pd.Timedelta(weeks=1)
earnings["date"] = earnings.date.dt.week + earnings.date.dt.year * 100
news["date"] = news["date"] - pd.Timedelta(weeks=1)
news["date"] = news.date.dt.week + news.date.dt.year * 100

full_tradingstrategy_ew = full_ts(flag_vw=False)
full_tradingstrategy_vw = full_ts(flag_vw=True)

full_no_earnings_ew = full_ts(False, 0, earnings, renaming="w/o Earnings")
full_no_earnings_vw = full_ts(True, 0, earnings, renaming="w/o Earnings")

full_earnings_ew = full_ts(False, 1, earnings, renaming="Earnings")
full_earnings_vw = full_ts(True, 1, earnings, renaming="Earnings")

full_no_news_ew = full_ts(False, 0, news, renaming="w/o News")
full_no_news_vw = full_ts(True, 0, news, renaming="w/o News")

full_news_ew = full_ts(False, 1, news, renaming="News")
full_news_vw = full_ts(True, 1, news, renaming="News")

output = pd.concat([full_tradingstrategy_ew, full_no_earnings_ew, full_earnings_ew])
output = pd.concat([output, full_tradingstrategy_vw, full_no_earnings_vw, full_earnings_vw])


to_plot = []

output = pd.concat([full_no_earnings_vw, full_earnings_vw, full_no_news_vw, full_news_vw])
output = output.reset_index().drop(columns="Weighting")
outperf = output.loc[output.Sample != "t-val"][[("", "N vs. L")]]

earnings_plot = output.loc[["Earnings" in x for x in output.Sample.values], :]
outperf_earnings = earnings_plot[[("", "N vs. L")]]
l_en = earnings_plot[["Sample", "L-En"]]
l_en.columns = ["Sample", "Pred", "Avg", "SD", "SR"]
l_en["model"] = "L-En"
n_en = earnings_plot[["Sample", "N-En"]]
n_en.columns = ["Sample", "Pred", "Avg", "SD", "SR"]
n_en["model"] = "N-En"
earnings_plot = pd.concat([l_en, n_en])
earnings_plot["Avg"] /= 100


news_plot = output.loc[["News" in x for x in output.Sample.values], :]
outperf_news = news_plot[[("", "N vs. L")]]
l_en = news_plot[["Sample", "L-En"]]
l_en.columns = ["Sample", "Pred", "Avg", "SD", "SR"]
l_en["model"] = "L-En"
n_en = news_plot[["Sample", "N-En"]]
n_en.columns = ["Sample", "Pred", "Avg", "SD", "SR"]
n_en["model"] = "N-En"
news_plot = pd.concat([l_en, n_en])
news_plot["Avg"] /= 100

fig, ax = plt.subplots(1, 2, figsize=(width, height), sharex=False, sharey=True)
to_plot = (earnings_plot, news_plot)
to_plot_perf = (outperf_earnings, outperf_news)
for i, dat in enumerate(to_plot):
    g = sns.barplot(x="Sample", y="Avg", hue="model", data=dat, palette=sns_palette(2), ax=ax[i])
    ax[i].set_xlabel("")
    ax[i].axhline(0, color="k", ls="--", lw=1, label="_nolegend_")
    if i == 0:
        ax[i].legend([], frameon=False)
    else:
        g.legend_.set_title(None)
        ax[i].legend(frameon=False, loc="upper right", ncol=1)
    ax[i].grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax[i].set_ylim([-0.006, 0.05])
    if i == 0:
        ax[i].set_ylabel("Realized Return")
    else:
        ax[i].set_ylabel("")
    ax[i].set_axisbelow(True)

    for j, (idx, val) in enumerate(to_plot_perf[i].iterrows()):
        ax[i].annotate(val.values[0], (j, -0.005), ha="center", color="black")
fig.savefig("../08_figures/returns_ea_news.pdf", bbox_inches="tight")


def latex_table(data, save_name: str, num_format="%.4f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    table = Tabular("".join(["l"] * len(data.index[0]) + ["r"] * (data.shape[1])), booktabs=True)
    table.add_row(
        [""] * len(data.index[0])
        + [
            MultiColumn(4, align="c", data=data.columns.get_level_values(0)[0]),
            "",
            MultiColumn(4, align="c", data=data.columns.get_level_values(0)[-3]),
            "",
            "",
        ]
    )
    table.add_hline(start=3, end=6)
    table.add_hline(start=8, end=11)
    to_add = data.index.names + ["" if "EMPTY" in c else c for c in data.columns.get_level_values(1).tolist()]
    to_add = [x if x != "Weighting" else "" for x in to_add]
    table.add_row([MultiColumn(1, align="c", data=c) for c in to_add])

    for i, (idx, row) in enumerate(data.iterrows()):
        to_add = []
        for _, num in row.iteritems():
            if isinstance(num, float):
                if np.isnan(num):
                    to_add.append("")
                else:

                    if idx[1] == "t-val":
                        num = "%.2f" % num
                        to_add.append(math(r"\footnotesize{(%s)}" % num))
                    else:
                        num = num_format % num
                        to_add.append(math(num))
            else:
                to_add.append(num)
        if i % 6 == 0:
            table.add_hline()
            table.add_row([MultiRow(6, data=idx[0])] + list(idx[1:]) + to_add)
        else:
            if idx[1] != "t-val":
                table.add_row([""] + list(idx[1:]) + to_add)
            else:
                table.add_row(["", ""] + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


latex_table(output, "trading_strat_earnings_weeklies", num_format="%.3f")
