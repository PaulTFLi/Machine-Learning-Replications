# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
import os
import pandas as pd
import numpy as np
from numba import njit

import glob

from scipy.stats import ttest_ind
from ff_reader import read_dataset
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

from pylatex import Tabular, MultiColumn, MultiRow, Math
from pylatex.utils import NoEscape

from analysis_setup import prediction_dict, analysisLoc, width, height, sns_palette
from get_python_matlab_dates import get_python_date_from_matlab_datenum


class_groups = pd.read_parquet(os.path.join(analysisLoc, "class_groups.pq"))
class_groups_with_info = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups_and_info.pq"),
    columns=[
        "date",
        "optionid",
        "permno",
        "ttm",
        "moneyness",
        "mve",
        "type",
        "delta",
        "gamma",
        "vega",
        "theta",
        "oi",
        "baspread",
        "optspread",
        "doi",
    ],
)
crsp = pd.read_parquet("../03_data/char_stocks/crsp_prices.pq", columns=["date", "permno", "close"])
crsp["close"] = np.abs(crsp["close"])
crsp["date"] = crsp["date"].dt.to_period("M")

class_groups_with_info = class_groups_with_info.merge(crsp, on=["date", "permno"])

# ---- OI market value
class_groups_with_info["oi"] *= class_groups_with_info["close"]
# ----

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


# %% Functions
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


@njit
def vw_mean(target, weight):
    return (target * weight).sum() / weight.sum()


# truncate:
def truncate(p, df):
    idx = (df > df.quantile(1 - p / 2)) | (df < df.quantile(p / 2))
    for col in idx:
        df.loc[idx[col], col] = np.nan
    return df


# winsorize
def winsorize(p, df):
    quantiles = df.quantile(1 - p / 2)
    idx = df > quantiles
    for col in idx:
        df.loc[idx[col], col] = quantiles[col]
    quantiles = df.quantile(p / 2)
    idx = df < quantiles
    for col in idx:
        df.loc[idx[col], col] = quantiles[col]
    return df


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

# %% Summary stats for options of different trading portfolios:
info_dict = {}
for model in ["L-En", "N-En"]:
    tmp = prediction_dict[model]["predictions"].copy().drop(columns=["permno"])
    tmp = tmp.merge(class_groups_with_info, on=["date", "optionid"])

    tmp["success"] = tmp["target"] > 0
    tmp["right_sign"] = np.sign(tmp["target"]) == np.sign(tmp["predicted"])

    # N_options = tmp.groupby(["date", "port"]).size().unstack().mean()
    # N_options.name = r"N^{OPT}"
    N_permno = (tmp.groupby(["date", "port"]).permno.nunique().unstack()).mean()
    N_permno.name = r"N^{U}"
    moneyness = tmp.groupby("port").moneyness.mean()
    ttm = tmp.groupby("port").ttm.mean()
    success = tmp.groupby("port").success.mean()
    right_sign = tmp.groupby("port").right_sign.mean()
    calls = (
        tmp[tmp["type"] == "call"].groupby(["date", "port"]).size().unstack()
        / tmp.groupby(["date", "port"]).size().unstack()
    ).mean()
    calls.name = r"\text{\% Calls}"
    spread = tmp.groupby("port").optspread.mean()
    oi = (
        tmp.groupby(["date", "port"])
        .oi.mean()
        .unstack()
        .divide(tmp.groupby(["date", "port"]).oi.mean().unstack().sum(axis=1), axis=0)
    ).mean()
    oi.name = r"\text{OI}"

    # calculate Greeks independent of underlying price: 20211012: done.
    greeks = tmp.groupby("port")[["delta", "gamma", "vega", "theta"]].mean()

    info = pd.concat(
        (N_permno, moneyness, ttm, success, right_sign, calls, spread, oi, greeks),
        axis=1,
    ).T
    info = info.rename(
        index={
            "moneyness": "m",
            "success": r"\hat{r}>0",
            "right_sign": r"s(r)=s(\hat{r})",
            "optspread": r"\text{Spread}",
        }
    )
    info.columns = [str(c + 1) for c in info.columns]
    info.columns = info.columns.str.replace("10", "Hi")
    info.columns = info.columns.str.replace("1", "Lo")

    info_dict[model] = info.copy()


def latex_table(data, save_name: str, num_format="%.4f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    table = Tabular("".join(["l"] + ["r"] * data.shape[1]), booktabs=True)
    table.add_row([""] + [MultiColumn(1, align="c", data=c) for c in data.columns])
    table.add_hline()

    for i, (idx, row) in enumerate(data.iterrows()):
        to_add = [math(idx)]
        for col, num in row.iteritems():
            if isinstance(num, float):
                if np.isnan(num):
                    to_add.append("")
                else:
                    to_add.append(math(num_format % num))
            else:
                to_add.append(num)
        table.add_row(to_add)

    table.generate_tex("../08_figures/%s" % save_name)


latex_table(info_dict["N-En"], "ts_summary_N-En", num_format="%.2f")
latex_table(info_dict["L-En"], "ts_summary_L-En", num_format="%.2f")

# %% Summary stats for options of different trading portfolios - Split by Put vs. Call:
model = "N-En"
tmp = prediction_dict[model]["predictions"].copy()
tmp = tmp.merge(class_groups_with_info, on=["date", "optionid"])


# % of calls over time
calls = (
    tmp[tmp["type"] == "call"].groupby(["date", "port"]).size().unstack()
    / tmp.groupby(["date", "port"]).size().unstack()
)
calls = calls.groupby(calls.index.year).mean()
calls = calls[[0, 2, 5, 6, 7, 9]]
calls.columns = [c + 1 for c in calls.columns]

fig, ax = plt.subplots(figsize=(width, height * 0.75))
colors = sns.color_palette("hls", calls.shape[1])
for i, col in enumerate(calls):
    ax.plot(calls[col], ls="--", marker="o", c=colors[i])
ax.legend(calls.columns, frameon=False, ncol=calls.shape[1], loc="upper right")
fig.tight_layout()
fig.savefig("../08_figures/ts_call_share.pdf")


# ttm and moneyness for calls vs. puts
moneyness = tmp.groupby(["port", "type"]).moneyness.mean().unstack()
ttm = tmp.groupby(["port", "type"]).ttm.mean().unstack()

fig, ax = plt.subplots(1, 2, figsize=(width, height * 0.75))
colors = ((1, 0, 0), (0, 0, 0))
for i, col in enumerate(moneyness):
    ax[0].plot(moneyness[col].index, moneyness[col].values, c=colors[i], marker="o", ls="")
    ax[0].set_title("$m$")
for i, col in enumerate(ttm):
    ax[1].plot(ttm[col].index, ttm[col].values, c=colors[i], marker="o", ls="")
    ax[1].set_title("ttm")
ax[1].legend(ttm.columns, frameon=False)
fig.savefig("../08_figures/ts_cp_mon_ttm.pdf")


# spreads for underlyings and option
spread = tmp.groupby(["port", "type"]).optspread.mean().unstack()
s_spread = tmp.groupby(["port", "type"]).baspread.mean().unstack()

fig, ax = plt.subplots(1, 2, figsize=(width, height * 0.75))
colors = ((1, 0, 0), (0, 0, 0))
for i, col in enumerate(spread):
    ax[0].plot(spread[col].index, spread[col].values, c=colors[i], marker="o", ls="")
    ax[0].set_title("Spread$^{OPT}$")
for i, col in enumerate(s_spread):
    ax[1].plot(s_spread[col].index, s_spread[col].values, c=colors[i], marker="o", ls="")
    ax[1].set_title("Spread$^{S}$")
ax[1].legend(s_spread.columns, frameon=False)
fig.savefig("../08_figures/ts_cp_spreads.pdf")


# greeks for calls vs. puts
greeks = tmp.groupby(["port", "type"])[["delta", "gamma", "vega", "theta"]].mean().unstack()

fig, ax = plt.subplots(2, 2, figsize=(width, height * 1.5))
colors = ((1, 0, 0), (0, 0, 0))
for i, col in enumerate(greeks.columns.get_level_values(0).unique()):
    x = i
    y = (i) % 2
    x -= y
    if i > 1:
        x -= 1
    print(x, y)
    for j, put_call in enumerate(greeks[col]):
        ax[x, y].plot(
            greeks[col][put_call].index,
            greeks[col][put_call].values,
            c=colors[j],
            marker="o",
            ls="",
        )
    ax[x, y].set_title(col)
ax[-1, -1].legend(s_spread.columns, frameon=False)
fig.savefig("../08_figures/ts_cp_greeks.pdf")


# %% Bucket transition matrix between portfolios:
model = "N-En"
tmp = prediction_dict[model]["predictions"].copy()
tmp = tmp.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])

transition = tmp.groupby(["date", "bucket", "permno", "port"]).size().unstack()  # get dist per permno
transition = transition.divide(transition.sum(axis=1), axis=0)  # get % dist per permno

# "mode" portfolios:
transition = transition.idxmax(axis=1)  # maximum per permno.

before = transition.groupby(["bucket", "permno"]).shift(1)
before.name = "From"
transition.name = "To"
transition = pd.concat((before, transition), axis=1).dropna()

transition = transition.groupby(["From", "To"]).size().unstack()
transition = transition.divide(transition.sum())
transition.columns = [str(int(c + 1)) for c in transition.columns]
transition.index = [str(int(c + 1)) for c in transition.index]

fig, ax = plt.subplots(figsize=(width, height * 0.75))
sns.heatmap(transition, ax=ax)
fig.savefig("../08_figures/ts_bucket_transition.pdf")


# %% Permno transition matrix between portfolios:
model = "N-En"
tmp = prediction_dict[model]["predictions"].copy()

transition = tmp.groupby(["date", "permno", "port"]).size().unstack()  # get dist per permno
transition = transition.divide(transition.sum(axis=1), axis=0)  # get % dist per permno

# "mode" portfolios:
transition = transition.idxmax(axis=1)  # maximum per permno.

before = transition.groupby("permno").shift()
before.name = "From"
transition.name = "To"
transition = pd.concat((before, transition), axis=1).dropna()

transition = transition.groupby(["From", "To"]).size().unstack()
transition = transition.divide(transition.sum())
transition.columns = [str(int(c + 1)) for c in transition.columns]
transition.index = [str(int(c + 1)) for c in transition.index]

fig, ax = plt.subplots(figsize=(width, height * 0.75))
sns.heatmap(transition, ax=ax)
fig.savefig("../08_figures/ts_permno_transition.pdf")

# %% Permno % in same portfolio over time:
model = "N-En"
tmp = prediction_dict[model]["predictions"].copy()
permno = tmp.groupby(["date", "permno", "port"]).size().unstack()
permno = permno.divide(permno.sum(axis=1), axis=0)
permno = permno.groupby("date").mean()
permno = permno.groupby(permno.index.year).mean()
permno = permno[[0, 2, 5, 6, 7, 9]]
permno.columns = [c + 1 for c in permno.columns]

fig, ax = plt.subplots(figsize=(width, height * 0.75))
colors = sns.color_palette("hls", permno.shape[1])
for i, col in enumerate(permno):
    ax.plot(permno[col], ls="--", marker="o", c=colors[i])
ax.legend(permno.columns, frameon=False, loc="upper right", ncol=permno.shape[1])
fig.tight_layout()
fig.savefig("../08_figures/ts_permno_same_portfolio.pdf")

# %% Trading strategy for the full sample
hml_returns = {}
to_plot = []
model_returns = {}
for model in ["L-En", "N-En"]:
    tmp = prediction_dict[model]["predictions"].copy()
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

ts_latex_table(full_sample, cp_sample, "trading_strat", num_format="%.3f")

# %% Trading strategy for the earnings vs non-earnings events
def full_ts(flag_vw=False, idx=None, additional=[], renaming="Full"):
    to_plot = []
    model_returns = {}
    for model in ["L-En", "N-En"]:
        tmp = prediction_dict[model]["predictions"].copy()
        if not (idx is None):
            name_idx = list(set(additional.columns) - set(["date", "permno"]))[0]
            tmp = tmp.reset_index()
            tmp = tmp.merge(additional, left_on=["date", "permno"], right_on=["date", "permno"], how="left")
            tmp[name_idx] = tmp[name_idx].fillna(0)
            tmp = tmp.loc[tmp[name_idx] == idx, :]
            tmp = tmp.drop(columns=[name_idx]).set_index("date")
        if flag_vw:
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

earnings = pd.read_parquet("../03_data/firm_level_events.pq", columns=["date", "permno", "earnings"])
earnings = earnings.dropna()
earnings["date"] = earnings.date.dt.to_period("M")

full_tradingstrategy_ew = full_ts(flag_vw=False)
full_tradingstrategy_vw = full_ts(flag_vw=True)

full_no_earnings_ew = full_ts(False, 0, earnings, renaming="w/o Earnings")
full_no_earnings_vw = full_ts(True, 0, earnings, renaming="w/o Earnings")

full_earnings_ew = full_ts(False, 1, earnings, renaming="Earnings")
full_earnings_vw = full_ts(True, 1, earnings, renaming="Earnings")

output = pd.concat([full_tradingstrategy_ew, full_no_earnings_ew, full_earnings_ew])
output = pd.concat([output, full_tradingstrategy_vw, full_no_earnings_vw, full_earnings_vw])

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

latex_table(output, "trading_strat_earnings", num_format="%.3f")

# %% Trading strategy using SENTIMENT:
vix = pd.read_csv("../03_data/sentiment/vix.csv", delimiter=";")
vix["date"] = pd.to_datetime(vix["date"], format="%d.%m.%Y")
vix = vix.set_index("date")
vix.index = vix.index.to_period("M")
vix = vix.groupby(vix.index).mean()
vix /= 100
vix = vix.loc["2003":"2020-11"]
vix = vix > vix.median()
vix.columns = ["idx"]

cfnai = pd.read_csv("../03_data/sentiment/CFNAI.csv")
cfnai["DATE"] = pd.to_datetime(cfnai["DATE"], format="%Y-%m-%d")
cfnai = cfnai.set_index("DATE")
cfnai.index.name = "date"
cfnai.index = cfnai.index.to_period("M")
cfnai = cfnai.groupby(cfnai.index).mean()
cfnai = cfnai.loc["2003":"2020-11"]
cfnai = cfnai > 0
cfnai.columns = ["idx"]

sentiment = pd.read_excel(
    "../03_data/sentiment/Investor_Sentiment_Data_20190327_POST.xlsx",
    engine="openpyxl",
    sheet_name="DATA",
).dropna(how="all")
sentiment["date"] = pd.to_datetime(sentiment["yearmo"].astype("int"), format="%Y%m")
sentiment = sentiment.set_index("date")
sentiment.index = sentiment.index.to_period("M")
sentiment = sentiment["SENT"]
sentiment = sentiment[sentiment != "."]
sentiment = sentiment.astype("f4")
sentiment = sentiment.to_frame()
sentiment = sentiment.groupby(sentiment.index).mean()
sentiment = sentiment.loc["2003":"2020-11"]
sentiment = sentiment > 0
sentiment.columns = ["idx"]

epu = pd.read_excel("../03_data/sentiment/US_Policy_Uncertainty_Data.xlsx", engine="openpyxl")
epu = epu.iloc[:-1]
epu["date"] = epu["Year"].astype("str") + epu["Month"].astype("int").astype("str").str.zfill(2)
epu["date"] = pd.to_datetime(epu["date"], format="%Y%m")
epu = epu.set_index("date")
epu.index = epu.index.to_period("M")
epu = epu[["News_Based_Policy_Uncert_Index"]]
epu = epu.groupby(epu.index).mean()
epu = epu.loc["2003":"2020-11"]
epu = epu > epu.median()
epu.columns = ["idx"]

stlfsi = pd.read_csv("../03_data/sentiment/STLFSI2.csv")
stlfsi["DATE"] = pd.to_datetime(stlfsi["DATE"], format="%Y-%m-%d")
stlfsi = stlfsi.set_index("DATE")
stlfsi.index.name = "date"
stlfsi.index = stlfsi.index.to_period("M")
stlfsi = stlfsi.groupby(stlfsi.index).mean()
stlfsi = stlfsi.loc["2003":"2020-11"]
stlfsi = stlfsi > 0
stlfsi.columns = ["idx"]


def sentiment_ts(idx, names):
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
            # returns = tmp.groupby(["date", "port"])[col].mean()
            returns = tmp[col].copy()
            returns = returns.unstack()
            returns.columns = "Lo 2 3 4 5 6 7 8 9 Hi".split()
            returns = returns["Hi"] - returns["Lo"]
            returns.name = col
            output.append(returns)
        output = pd.concat(output, axis=1)
        output = output.merge(idx, on="date")
        model_returns[model] = output["target"].copy()

        grouper = output.groupby(["idx"])
        strat = grouper.mean()
        strat.columns = ["Pred", "Avg"]
        strat["SD"] = grouper.target.std().tolist()
        strat *= 100
        strat["SR"] = strat["Avg"] / strat["SD"]
        strat.columns = pd.MultiIndex.from_arrays([[model] * strat.shape[1], strat.columns])
        tvals = [
            grouper.apply(
                lambda x: sm.OLS(x["target"], np.ones(x["target"].shape))
                .fit(cov_type="HAC", cov_kwds={"maxlags": 12})
                .tvalues
            )["const"],
            grouper.apply(lambda x: Lo_tval_SR(x["target"])),
        ]
        tvals = pd.concat(tvals, axis=1)
        tvals.insert(0, "EMPTY_1", "")
        tvals.insert(2, "EMPTY_2", "")
        tvals.columns = strat.columns
        strat = pd.concat((strat, tvals))
        strat = strat.sort_index()
        to_plot.append(strat)

    model_returns = pd.concat(model_returns, axis=1)

    to_plot = pd.concat(to_plot, axis=1)
    to_plot.insert(4, "EMPTY_0", "")
    to_plot["EMPTY_1"] = ""

    model_returns = model_returns.merge(idx, on="date")
    val = model_returns.groupby("idx").apply(lambda x: ttest_ind(x["N-En"], x["L-En"])[0]).values
    val = np.array([val[0], 0, val[1], 0])
    to_plot[("", "N vs. L")] = ""
    to_plot.loc[np.abs(val) > 1.65, ("", "N vs. L")] = "*"
    to_plot.loc[np.abs(val) > 1.96, ("", "N vs. L")] = "**"
    to_plot.loc[np.abs(val) > 2.56, ("", "N vs. L")] = "***"

    to_plot = to_plot.rename(index={False: names[0], True: names[1]})
    to_plot.index = np.hstack([[c, "t-val"] for c in to_plot.index[::2].tolist()])
    return to_plot


output = sentiment_ts(vix, ["Low VIX", "High VIX"])
empty_line = pd.DataFrame(data="", columns=output.columns, index=["EMPTY"])
output = pd.concat((output, empty_line, sentiment_ts(epu, ["Low EPU", "High EPU"])))
output = pd.concat((output, empty_line, sentiment_ts(cfnai, ["Neg. CFNAI", "Pos. CFNAI"])))
output = pd.concat((output, empty_line, sentiment_ts(stlfsi, ["Low FED Stress", "High FED Stress"])))
output = pd.concat((output, empty_line, sentiment_ts(sentiment, ["Low SENT", "High SENT"])))


def latex_table(data, save_name: str, num_format="%.4f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    table = Tabular("".join(["l"] + ["c"] * (data.shape[1] - 1) + ["c"]), booktabs=True)
    table.add_row(
        [
            "",
            MultiColumn(4, align="c", data=data.columns.get_level_values(0)[0]),
            "",
            MultiColumn(4, align="c", data=data.columns.get_level_values(0)[-3]),
            "",
            "",
        ]
    )
    to_add = [""] + ["" if "EMPTY" in c else c for c in data.columns.get_level_values(1).tolist()]
    table.add_row([MultiColumn(1, align="c", data=c) for c in to_add])
    table.add_hline(start=2, end=5)
    table.add_hline(start=7, end=10)
    table.add_hline(start=12, end=12)

    for _, (idx, row) in enumerate(data.iterrows()):
        to_add = []
        if idx == "EMPTY":
            table.add_hline()
            table.add_empty_row()
        else:
            for col, num in row.iteritems():
                if isinstance(num, float):
                    if np.isnan(num):
                        to_add.append("")
                    else:
                        if idx == "t-val":
                            num = "%.2f" % num
                            to_add.append(math(r"\footnotesize{(%s)}" % num))
                        else:
                            num = num_format % num
                            to_add.append(math(num))
                else:
                    to_add.append(num)
            if idx != "t-val":
                table.add_row([idx] + to_add)
            else:
                table.add_row([""] + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


latex_table(output, "trading_strat_sentiment", num_format="%.3f")


# %% Trading strategy using **value-weights*:
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
hml_returns["full"] = model_returns.copy()

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

ts_latex_table(full_sample, cp_sample, "trading_strat_vw", num_format="%.3f")

# %%
# Trading strategy using **value-weights* ALLOW JUST ONE OPTION PER UNDERLYING
to_plot = []
model_returns = {}
for model in ["L-En", "N-En"]:
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()
    tmp["predicted_abs"] = tmp.predicted.abs()

    tmp = tmp.reset_index()
    largest = tmp.loc[tmp.groupby(["date", "permno"]).predicted_abs.idxmax().values]
    tmp = largest.set_index("date")
    tmp = tmp.drop_duplicates()

    tmp["port"] = tmp.groupby("date").predicted.transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))

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
# hml_returns["full"] = model_returns.copy()

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
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()
    tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])
    tmp["predicted_abs"] = tmp.predicted.abs()

    tmp = tmp.reset_index()
    largest = tmp.loc[tmp.groupby(["date", "permno", col]).predicted_abs.idxmax().values]
    tmp = largest.set_index("date")
    tmp = tmp.drop_duplicates()

    tmp["port"] = tmp.groupby(["date", col]).predicted.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )

    # ---- DOI-weighted portfolios, across underlyings
    tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi"]], on=["date", "optionid"])
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


ts_latex_table(full_sample, cp_sample, "trading_strat_vw_one_option_per_permno", num_format="%.3f")

# %%
# Trading strategy using **value-weights* SPLIT OPTIONS IN DECILES FOR EACH PERMNO
to_plot = []
model_returns = {}
for model in ["L-En", "N-En"]:
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()
    tmp = tmp.drop(columns="port")
    tmp = tmp.reset_index()
    # tmp = tmp.groupby(["date", "permno"]).filter(lambda x: x.shape[0]>=10) # at least one option per decile
    tmp["port"] = tmp.groupby(["date", "permno"]).predicted.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    tmp = tmp.set_index("date")
    tmp = tmp.drop_duplicates()

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
# hml_returns["full"] = model_returns.copy()

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
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()
    tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])

    tmp = tmp.reset_index()
    # tmp = tmp.groupby(["date", "permno",col]).filter(lambda x: x.shape[0]>=10) # at least one option per decile
    tmp["port"] = tmp.groupby(["date", "permno", col]).predicted.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    tmp = tmp.set_index("date")
    tmp = tmp.drop_duplicates()

    tmp["port"] = tmp.groupby(["date", col]).predicted.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )

    # ---- DOI-weighted portfolios, across underlyings
    tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi"]], on=["date", "optionid"])
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

ts_latex_table(full_sample, cp_sample, "trading_strat_vw_all_options_per_permno_split", num_format="%.3f")

# %%
# Trading strategy using **value-weights* EACH PERMNO MUST HAVE ONE OPTION PER DECILE (NOT MORE)
to_plot = []
model_returns = {}
for model in ["L-En", "N-En"]:
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()
    tmp = tmp.drop(columns="port")
    tmp = tmp.reset_index()
    tmp["predicted_abs"] = tmp.predicted.abs()
    # tmp = tmp.groupby(["date", "permno"]).filter(lambda x: x.shape[0]>=10) # at least one option per decile
    tmp["port"] = tmp.groupby(["date", "permno"]).predicted.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )

    tmp = tmp.loc[tmp.groupby(["date", "permno", "port"]).predicted_abs.idxmax().values]
    tmp = tmp.set_index("date")
    tmp = tmp.drop_duplicates()
    tmp = tmp.drop(columns="predicted_abs")

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
# hml_returns["full"] = model_returns.copy()

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
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()
    tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])

    tmp = tmp.reset_index()
    # tmp = tmp.groupby(["date", "permno",col]).filter(lambda x: x.shape[0]>=10) # at least one option per decile
    tmp["port"] = tmp.groupby(["date", "permno", col]).predicted.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    tmp["predicted_abs"] = tmp.predicted.abs()
    tmp = tmp.loc[tmp.groupby(["date", "permno", "port", col]).predicted_abs.idxmax().values]
    tmp = tmp.set_index("date")
    tmp = tmp.drop_duplicates()
    tmp = tmp.drop(columns="predicted_abs")

    tmp["port"] = tmp.groupby(["date", col]).predicted.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )

    # ---- DOI-weighted portfolios, across underlyings
    tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi"]], on=["date", "optionid"])
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

ts_latex_table(full_sample, cp_sample, "trading_strat_vw_each_decile_one_option_permno", num_format="%.3f")

# %%
# Trading strategy using **value-weights* ALLOW JUST TWO OPTIONS PER UNDERLYING
to_plot = []
model_returns = {}
for model in ["L-En", "N-En"]:
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()

    tmp = tmp.reset_index()
    largest = tmp.loc[tmp.groupby(["date", "permno"]).predicted.idxmax().values]
    smallest = tmp.loc[tmp.groupby(["date", "permno"]).predicted.idxmin().values]
    tmp = pd.concat((largest, smallest))
    tmp = tmp.set_index("date")
    tmp = tmp.drop_duplicates()

    tmp["port"] = tmp.groupby("date").predicted.transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))

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
# hml_returns["full"] = model_returns.copy()

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
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()
    tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])

    tmp = tmp.reset_index()
    largest = tmp.loc[tmp.groupby(["date", "permno", col]).predicted.idxmax().values]
    smallest = tmp.loc[tmp.groupby(["date", "permno", col]).predicted.idxmin().values]
    tmp = pd.concat((largest, smallest))
    tmp = tmp.set_index("date")
    tmp = tmp.drop_duplicates()

    tmp["port"] = tmp.groupby(["date", col]).predicted.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )

    # ---- DOI-weighted portfolios, across underlyings
    tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi"]], on=["date", "optionid"])
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

ts_latex_table(full_sample, cp_sample, "trading_strat_vw_two_option_per_permno", num_format="%.3f")

# %%
# Trading strategy with **value_weights** REQUIRING ALL OPTIONS OF A PERMNO TO BE USED IN ONE DECILE
to_plot = []
model_returns = {}
for model in ["L-En", "N-En"]:
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()

    average_expected_return = tmp.groupby(["date", "permno"])["predicted"].mean()
    ports = average_expected_return.groupby("date").transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
    ports.name = "port"
    tmp = tmp.drop(columns=["port"]).merge(ports, on=["date", "permno"])

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
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()
    tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])

    tmp = tmp.reset_index()

    average_expected_return = tmp.groupby(["date", "permno", col])["predicted"].mean()
    ports = average_expected_return.groupby(["date", col]).transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    ports.name = "port"
    tmp = tmp.drop(columns=["port"]).merge(ports, on=["date", "permno"])

    tmp = tmp.set_index("date")
    tmp = tmp.drop_duplicates()

    # ---- DOI-weighted portfolios, across underlyings
    tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi"]], on=["date", "optionid"])
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

ts_latex_table(full_sample, cp_sample, "trading_strat_vw_all_options_of_permno", num_format="%.3f")

# %% Trading strategy per option bucket
to_plot = []
model_returns = {}
col = "bucket"
for model in ["L-En", "N-En"]:
    print(f"Working on model {model}")
    tmp = prediction_dict[model]["predictions"].copy()
    tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])
    tmp = tmp[~tmp.bucket.str.contains("nan")]

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

    total_weights = tmp.groupby(["date", "bucket", "port"]).doi.sum()  # denominator
    total_weights.name = "summed_weights"
    tmp = tmp.groupby(["date", "bucket", "port"]).sum()  # numerator
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
hml_returns["buckets"] = model_returns.copy()

to_plot = pd.concat(to_plot, axis=1)

val = model_returns.groupby(["bucket", "port"]).apply(lambda x: ttest_ind(x["N-En"], x["L-En"])[0])
to_plot[("", "N vs. L")] = ""
to_plot.loc[val.abs() > 1.65, ("", "N vs. L")] = "*"
to_plot.loc[val.abs() > 1.96, ("", "N vs. L")] = "**"
to_plot.loc[val.abs() > 2.56, ("", "N vs. L")] = "***"
to_plot = to_plot.reset_index()

ttm = []
moneyness = []
types = []
for i, bucket in enumerate(to_plot.bucket.str.split("_").values):
    if "long" in bucket:
        ttm.append(r"\tau > 90")
    else:
        ttm.append(r"\tau \leq 90")
    if "itm" in bucket:
        moneyness.append("itm")
    elif "atm" in bucket:
        moneyness.append("atm")
    elif "otm" in bucket:
        moneyness.append("otm")
    if "call" in bucket:
        types.append("C")
    elif "put" in bucket:
        types.append("P")
    else:
        types.append("")
to_plot["TTM"] = ttm
to_plot["Mon."] = moneyness
to_plot["Type"] = types

to_plot = to_plot.drop(columns=["bucket", "port"])
to_plot = to_plot.sort_values(["TTM", "Mon.", "Type"], ascending=[False, True, True])
to_plot = to_plot.set_index(["TTM", "Mon.", "Type"])
to_plot.insert(4, "EMPTY_0", "")
to_plot.insert(9, "EMPTY_1", "")


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
    table.add_hline(start=4, end=7)
    table.add_hline(start=9, end=12)
    to_add = data.index.names + ["" if "EMPTY" in c else c for c in data.columns.get_level_values(1).tolist()]
    table.add_row([MultiColumn(1, align="c", data=c) for c in to_add])

    for i, (idx, row) in enumerate(data.iterrows()):
        to_add = []
        for _, num in row.iteritems():
            if isinstance(num, float):
                if np.isnan(num):
                    to_add.append("")
                else:
                    to_add.append(math(num_format % num))
            else:
                to_add.append(num)
        if i % 5 == 0:
            table.add_hline()
            table.add_row([MultiRow(5, data=math(idx[0]))] + list(idx[1:]) + to_add)
        else:
            table.add_row([""] + list(idx[1:]) + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


latex_table(to_plot, "trading_strat_bucket", num_format="%.3f")

# %% Explaining trading strat alphas with many factor models:

ff = read_dataset("F-F_Research_Data_5_Factors_2x3")
mom = read_dataset("F-F_Momentum_Factor")
mom.columns = ["MOM"]
ff = ff.merge(mom, left_index=True, right_index=True)
ff = ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]]
ff = ff.shift(-1)

ps_liq = pd.read_csv("../03_data/ps_liq.csv", sep=";", usecols=["Month", "LIQ_V"])
ps_liq["Month"] = pd.to_datetime(ps_liq["Month"], format="%Y%m").dt.to_period("M")
ps_liq.set_index("Month", inplace=True)
ps_liq = ps_liq.shift(-1)
ps_liq = ps_liq[ps_liq["LIQ_V"] != -99]
ff_ps = pd.merge(ff, ps_liq, left_index=True, right_index=True)

stambaugh = pd.read_csv("../03_data/stambaugh.csv")
stambaugh["YYYYMM"] = pd.to_datetime(stambaugh["YYYYMM"], format="%Y%m").dt.to_period("M")
stambaugh = stambaugh.set_index("YYYYMM")
stambaugh = stambaugh.drop(columns=["RF"])
stambaugh = stambaugh.shift(-1)

bali = pd.read_excel("../03_data/OPT Factors.xlsx", engine="openpyxl")
bali["date"] = pd.to_datetime(bali["date"], format="%Y%m")
bali = bali.set_index("date")
bali.index = bali.index.to_period("M")
bali = bali.iloc[:, :3]
bali = bali.shift(-1)
bali = bali.merge(ff["Mkt-RF"], left_index=True, right_index=True)

agarwal = pd.read_excel("../03_data/optionreturns_0283_0519.xlsx", engine="openpyxl")
agarwal["Date"] = pd.to_datetime(agarwal["Date"], format="%Y-%m-%d")
agarwal = agarwal.set_index("Date")
agarwal.index = agarwal.index.to_period("M")
agarwal = agarwal.iloc[:, :4]
agarwal = agarwal.shift(-1)
agarwal = agarwal.merge(ff["Mkt-RF"], left_index=True, right_index=True)

lbc = pd.read_csv("../03_data/lbc_monthly.csv")
lbc["date"] = get_python_date_from_matlab_datenum(lbc.date.astype("f4").to_numpy())["Python Date"]
lbc = lbc.set_index("date")
lbc.index = lbc.index.to_period("M")
lbc = lbc.shift(-1)
lbc = lbc.merge(ff["Mkt-RF"], left_index=True, right_index=True)


def factor_model(data, factors, data_col: str, name: str, only_hml: bool = False):
    factor_cols = factors.columns
    data = data.merge(factors, left_on="date", right_index=True)
    if "bucket" in data.index.names:
        reg = data.groupby(["bucket", "port"]).apply(
            lambda x: sm.OLS(x[data_col], sm.add_constant(x[factor_cols]), missing="drop").fit(
                cov_type="HAC", cov_kwds={"maxlags": 12}
            )
        )
    else:
        reg = data.groupby(["port"]).apply(
            lambda x: sm.OLS(x[data_col], sm.add_constant(x[factor_cols]), missing="drop").fit(
                cov_type="HAC", cov_kwds={"maxlags": 12}
            )
        )
    params = reg.apply(lambda x: x.params.const)
    params.name = "params"
    tvalues = reg.apply(lambda x: x.tvalues.const)
    tvalues.name = "tvalues"
    joined = pd.concat((params, tvalues), axis=1)
    if only_hml:
        joined = joined[joined.index.get_level_values("port") == "H-L"]
    joined = joined.reset_index()

    if "bucket" in data.index.names:
        ttm = []
        moneyness = []
        types = []
        for _, bucket in enumerate(joined.bucket.str.split("_").values):
            if "long" in bucket:
                ttm.append(r"\tau > 90")
            else:
                ttm.append(r"\tau \leq 90")
            if "itm" in bucket:
                moneyness.append("itm")
            elif "atm" in bucket:
                moneyness.append("atm")
            elif "otm" in bucket:
                moneyness.append("otm")
            if "call" in bucket:
                types.append("C")
            elif "put" in bucket:
                types.append("P")
            else:
                types.append("")
        joined["TTM"] = ttm
        joined["Mon."] = moneyness
        joined["Type"] = types

        joined = joined.drop(columns=["bucket", "port"])
        joined = joined.sort_values(["TTM", "Mon.", "Type"], ascending=[False, True, True])
        joined = joined.set_index(["TTM", "Mon.", "Type"])

    return joined


alphas = {}
tvalues = {}
for key in hml_returns:
    # ---- Raw returns
    raw = factor_model(hml_returns[key].copy(), sm.add_constant(ff)[["const"]], "N-En", "CAPM", True)

    # ---- CAPM
    capm = factor_model(hml_returns[key].copy(), ff[["Mkt-RF"]], "N-En", "CAPM", True)

    # ---- FF6
    ff6 = factor_model(hml_returns[key].copy(), ff, "N-En", "FF6", True)

    # ---- FF6 + PS
    ff6ps = factor_model(hml_returns[key].copy(), ff_ps, "N-En", "FF6+PS", True)

    # ---- Stambaugh/Yuan
    # sy = factor_model(hml_returns[key].copy(), stambaugh, "N-En", "SY", True)

    # ---- Agarwal/Naik
    an = factor_model(hml_returns[key].copy(), agarwal, "N-En", "AN", True)

    # ---- Bali
    b_mod = factor_model(hml_returns[key].copy(), bali, "N-En", "BM", True)

    # ---- LBC
    lbc_mod = factor_model(hml_returns[key].copy(), lbc, "N-En", "LBC", True)

    alphas[key] = pd.concat((capm.params, ff6.params, ff6ps.params, an.params, b_mod.params, lbc_mod.params), axis=1)
    alphas[key].columns = ["CAPM", "FF6", "FF6+PS", "AN", "BM", "LBC"]
    tvalues[key] = pd.concat(
        (capm.tvalues, ff6.tvalues, ff6ps.tvalues, an.tvalues, b_mod.tvalues, lbc_mod.tvalues),
        axis=1,
    )
    tvalues[key].columns = ["CAPM", "FF6", "FF6+PS", "AN", "BM", "LBC"]


def latex_table_long(data_full, tval_full, data_bucket, tval_bucket, save_name: str, num_format="%.4f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    N = len(data_bucket.index[0]) + data_full.shape[1]
    table = Tabular("".join(["l"] * len(data_bucket.index[0]) + ["c"] * (data_full.shape[1])), booktabs=True)
    table.add_row([""] * len(data_bucket.index[0]) + [MultiColumn(1, align="c", data=c) for c in data_full.columns])
    table.add_hline(start=2, end=N)

    # ---- full data:
    for i, (idx, row) in enumerate(data_full.iterrows()):
        # params
        to_add = []
        for col, num in row.iteritems():
            if isinstance(num, float):
                if np.isnan(num):
                    to_add.append("")
                else:
                    to_add.append(math(num_format % num))
            else:
                to_add.append(num)
        table.add_row([MultiColumn(3, align="r", data="Full Sample")] + to_add)

        # tvalues
        to_add = []
        for col, num in tval_full.iloc[i].iteritems():
            if isinstance(num, float):
                if np.isnan(num):
                    to_add.append("")
                else:
                    num = num_format % num
                    to_add.append(math(r"(%s)" % num))
            else:
                to_add.append(num)
        table.add_row([""] * len(data_bucket.index[0]) + to_add)

    # ---- buckets:
    table.add_hline()
    table.add_empty_row()
    table.add_row((MultiColumn(N, align="c", data="Buckets"),))
    table.add_row([""] * len(data_bucket.index[0]) + [MultiColumn(1, align="c", data=c) for c in data_full.columns])
    for i, (idx, row) in enumerate(data_bucket.iterrows()):
        to_add = []
        # params
        for _, num in row.iteritems():
            if isinstance(num, float):
                if np.isnan(num):
                    to_add.append("")
                else:
                    to_add.append(math(num_format % num))
            else:
                to_add.append(num)
        if i % 5 == 0:
            table.add_hline()
            table.add_row([MultiRow(10, data=idx[0])] + list(idx[1:]) + to_add)
        else:
            table.add_row([""] + list(idx[1:]) + to_add)

        # tvalues
        to_add = []
        for col, num in tval_bucket.iloc[i].iteritems():
            if isinstance(num, float):
                if np.isnan(num):
                    to_add.append("")
                else:
                    num = num_format % num
                    to_add.append(math(r"(%s)" % num))
            else:
                to_add.append(num)
        table.add_row([""] * len(data_bucket.index[0]) + to_add)

    table.generate_tex("../08_figures/%s" % save_name)

def latex_table_wide(
    data_full,
    tval_full,
    data_bucket,
    tval_bucket,
    save_name: str,
    num_format="%.4f",
    tval_format="%.4f",
    split_at_n_models=3,
):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    num_cols = min(split_at_n_models, data_full.shape[1])
    loopers = range(0, data_full.shape[1], 3)
    table = Tabular(
        "".join(["l"] * len(data_bucket.index[0]) + ["c"] * (num_cols * 3)),
        booktabs=True,
    )
    for i_n, n in enumerate(loopers):
        N = len(data_bucket.index[0]) + num_cols * 3
        table.add_row(
            [""] * len(data_bucket.index[0])
            + [MultiColumn(3, align="c", data=c) for c in data_full.columns[n : n + num_cols]]
        )
        table.add_hline(start=1, end=N)

        # ---- full data:
        for i, (_, row) in enumerate(data_full.iloc[:, n : n + num_cols].iterrows()):
            to_add = []
            # params
            for i_num, num in row.iteritems():
                to_add.append(math(num_format % num))
                num = tval_full.iloc[i][i_num]
                num = tval_format % num
                to_add.append(math(r"(%s)" % num))
                to_add.append("")
            # tvalues
            table.add_row([MultiColumn(3, align="l", data="Full Sample")] + to_add)

        # ---- buckets:
        # table.add_hline(start=1, end=N)
        table.add_empty_row()
        table.add_row((MultiColumn(N, align="c", data="Buckets"),))
        for i, (idx, row) in enumerate(data_bucket.iloc[:, n : n + num_cols].iterrows()):
            to_add = []
            # params
            for i_num, num in row.iteritems():
                to_add.append(math(num_format % num))
                num = tval_bucket.iloc[i][i_num]
                num = tval_format % num
                to_add.append(math(r"(%s)" % num))
                to_add.append("")
            if i % 5 == 0:
                if i > 0:
                    table.add_hline(start=1, end=N)
                else:
                    table.add_hline(start=1, end=N)
                table.add_row([MultiRow(5, data=math(idx[0]))] + list(idx[1:]) + to_add)
            else:
                table.add_row([""] + list(idx[1:]) + to_add)

        if i_n < (len(loopers) - 1):
            table.add_empty_row()
            table.add_empty_row()
            table.add_hline()

    table.generate_tex("../08_figures/%s" % save_name)


latex_table_wide(
    alphas["full"] * 100,
    tvalues["full"],
    alphas["buckets"] * 100,
    tvalues["buckets"],
    "trading_strat_alphas_wide",
    num_format="%.3f",
    tval_format="%.2f",
)


# %%
# Trading strategy using **value-weights* OVER TIME:
to_plot = []
model_returns = {}
for model in ["N-En"]:
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
        returns.name = "return"
        returns.index.names = ["date", "port"]
        returns = returns.to_frame()
        returns["type"] = col
        output.append(returns)
    output = pd.concat(output, axis=0)
    output = output[output.index.get_level_values("port") == "H-L"]
    output = output.reset_index(level="port", drop=True)
    # output = output.groupby([pd.Grouper(freq="A"), "type"]).mean()

    means = output.groupby([pd.Grouper(freq="A"), "type"]).mean()
    stds = output.groupby([pd.Grouper(freq="A"), "type"]).std()
    out = means / stds
    out.columns = ["Realized SR"]
    means.columns = ["Returns"]
    out = pd.concat((out, means), axis=1)
    out = out.reset_index()

out["type"] = out["type"].replace({"predicted": "E[r]", "target": "r"})
fig, ax = plt.subplots(2, 1, figsize=(width, height * 0.8), dpi=800, sharex=True)
sns.barplot(data=out, x="date", y="Returns", hue="type", ax=ax[0], palette=sns_palette(2))
ax[0].legend(frameon=False, ncol=2)
sns.barplot(data=out[out["type"] == "r"], x="date", y="Realized SR", ax=ax[1], color=sns_palette(2)[1])
ax[0].set_xlabel("")
ax[1].set_xlabel("")
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30)
ax[0].grid(axis="y", ls="--", lw=0.5, color="k")
ax[0].set_axisbelow(True)
ax[1].grid(axis="y", ls="--", lw=0.5, color="k")
ax[1].set_axisbelow(True)
fig.tight_layout()
fig.savefig("../08_figures/ts_over_time.pdf")


# %%
# TS over time using value-weights SPLIT by PUTS and CALLS
to_plot = []
model_returns = {}
col = "type"
for model in ["N-En"]:
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
    for col in ["predicted", "target"]:
        returns = tmp[col].unstack()
        returns.columns = "Lo 2 3 4 5 6 7 8 9 Hi".split()
        returns["H-L"] = returns["Hi"] - returns["Lo"]
        returns = returns.stack()
        returns.name = "return"
        returns.index.names = ["date", "type", "port"]
        returns = returns.to_frame()
        returns["style"] = col
        output.append(returns)
    output = pd.concat(output, axis=0)
    output = output[output.index.get_level_values("port") == "H-L"]
    output = output.reset_index(level=["port"], drop=True)
    output = output.reset_index(level=["type"])
    # output = output.groupby([pd.Grouper(freq="A"), "type"]).mean()

    means = output.groupby([pd.Grouper(freq="A"), "type", "style"]).mean()
    stds = output.groupby([pd.Grouper(freq="A"), "type", "style"]).std()
    out = means / stds
    out.columns = ["Realized SR"]
    means.columns = ["Returns"]
    out = pd.concat((out, means), axis=1)
    out = out.reset_index()

out["style"] = out["style"].replace({"predicted": "E[r]", "target": "r"})

fig, ax = plt.subplots(2, 2, figsize=(width, height * 0.8), dpi=800, sharex=True, sharey="col")
# calls
sns.barplot(data=out[out["type"] == "call"], x="date", y="Returns", hue="style", ax=ax[0, 0], palette=sns_palette(2))
sns.barplot(
    data=out[(out["style"] == "r") & (out["type"] == "call")],
    x="date",
    y="Realized SR",
    ax=ax[0, 1],
    color=sns_palette(2)[1],
)
# puts
sns.barplot(data=out[out["type"] == "put"], x="date", y="Returns", hue="style", ax=ax[1, 0], palette=sns_palette(2))
sns.barplot(
    data=out[(out["style"] == "r") & (out["type"] == "put")],
    x="date",
    y="Realized SR",
    ax=ax[1, 1],
    color=sns_palette(2)[1],
)
# plot stuff
ax[0, 0].legend(frameon=False, ncol=2)
ax[1, 0].legend([], frameon=False, ncol=2)
ax[0, 1].legend([], frameon=False, ncol=2)
ax[1, 1].legend([], frameon=False, ncol=2)
ax[0, 0].set_ylabel("Calls")
ax[0, 1].set_ylabel("")
ax[1, 0].set_ylabel("Puts")
ax[1, 1].set_ylabel("")
ax[0, 0].set_xlabel("")
ax[0, 1].set_xlabel("")
ax[1, 0].set_xlabel("")
ax[1, 1].set_xlabel("")
ax[0, 0].set_title("Returns")
ax[0, 1].set_title("Realized SR")
ax[1, 0].set_xticklabels(ax[1, 0].get_xticklabels(), rotation=45)
ax[1, 1].set_xticklabels(ax[1, 1].get_xticklabels(), rotation=45)
ax[0, 0].grid(axis="y", ls="--", lw=0.5, color="k")
ax[0, 1].grid(axis="y", ls="--", lw=0.5, color="k")
ax[1, 0].grid(axis="y", ls="--", lw=0.5, color="k")
ax[1, 1].grid(axis="y", ls="--", lw=0.5, color="k")
ax[0, 0].set_axisbelow(True)
ax[0, 1].set_axisbelow(True)
ax[1, 0].set_axisbelow(True)
ax[1, 1].set_axisbelow(True)
fig.tight_layout()
fig.savefig("../08_figures/ts_over_time_cp.pdf")


# %% Trading strategy: impact of transaction costs
hml_returns = {}
n_ports = 10
spread_returns = pd.read_parquet("../03_data/returns_spreads.pq")
# spread_returns = pd.read_parquet("../03_data/returns_spreads_init.pq")
spread_returns = spread_returns.reset_index()
spread_returns["optionid"] = spread_returns["optionid"].astype("int")
spread_returns["date"] = spread_returns["date"].dt.to_period("M")
spread_returns = spread_returns.set_index(["date", "optionid"])

# get optspread relative to denominator and check whether option matures in considered month
columns_classifier = [
    "date",
    "optionid",
    "doi",
    "optspread",
    "S_spreads_paid",
    "ttm",
    "mid",
    "gain_dh_daily",
    "return_dh_daily_inv",
    "baspread",
    "delta",
    "margin_denominator",
    "margin_denominator_signed",
]
classifier = pd.concat(
    [pd.read_parquet(f, columns=columns_classifier) for f in glob.glob("../04_results/option_sample/classifier*.pq")]
)
classifier["denominator"] = classifier.gain_dh_daily / classifier.return_dh_daily_inv
classifier["optspread"] = classifier.optspread * classifier.mid  # absolute option spread
classifier["non_exp_month"] = (classifier.ttm > 30).astype(int)
classifier["rel_optspread"] = classifier["optspread"] / 2 / classifier["denominator"]
classifier["rel_S_spreads_paid"] = classifier.S_spreads_paid / 2 / classifier["denominator"]
classifier["exp_S_spreads_paid"] = classifier.baspread / 2 * 3 * classifier.delta.abs() / classifier["denominator"]

# %%
# Game spreads by considering not just E[r] but E[r] minus spreads
levels_spreads = [0.25, 0.5, 0.75, 1]

# only transaction costs for option prices
to_plot = []
model_returns = {}

for model in ["L-En", "N-En"]:
    print(model)
    data = prediction_dict[model]["predictions"].copy()
    data = data.set_index(["optionid"], append=True)
    data = data[["target", "predicted"]]
    data.columns = ["spread=0", "predicted"]
    data["spread=0_short"] = data["spread=0"]
    data = data.merge(spread_returns, on=["date", "optionid"], how="left")

    # ---- DOI-weighted portfolios, across underlyings
    data = data.merge(
        classifier[["date", "optionid", "doi", "rel_optspread", "non_exp_month"]], on=["date", "optionid"], how="left"
    )
    # data = data.drop_duplicates()  # to be safe.

    out_df = []
    out_df_sr = []
    out_df_tvals = []
    for level_spread in levels_spreads:
        print(level_spread)
        # twice the spread as expected costs if option does not mature in considered month; otherwise just once the spread as E[costs]
        spread_costs = (data["rel_optspread"] * level_spread * (1 + data["non_exp_month"])).to_numpy()
        tmp = data.copy()
        tmp.loc[tmp.predicted >= 0, "predicted"] = np.maximum(
            0, tmp.loc[tmp.predicted >= 0, "predicted"] - spread_costs[tmp.predicted >= 0]
        )
        tmp.loc[tmp.predicted < 0, "predicted"] = np.minimum(
            0, tmp.loc[tmp.predicted < 0, "predicted"] + spread_costs[tmp.predicted < 0]
        )

        # test
        tmp = tmp.loc[tmp.predicted != 0, :]
        # test end

        tmp["port"] = tmp.groupby("date").predicted.transform(
            lambda x: pd.qcut(x, n_ports, labels=False, duplicates="drop")
        )
        tmp = tmp[tmp["port"].isin([0, 9])]
        tmp = tmp.set_index(["date", "optionid"])
        tmp = tmp.drop(columns=["predicted"])
        tmp = tmp.drop(
            columns=[c for c in tmp.columns if "=" + str(int(level_spread * 100)) not in c and c.startswith("spread")]
        )

        weighted = tmp[[c for c in tmp.columns if c.startswith("spread=")]].multiply(tmp["doi"], axis=0)
        tmp[weighted.columns] = weighted

        # needed if we truncated, since some DOI are "missing" if the return is "missing"
        for col in [c for c in tmp.columns if c.startswith("spread")]:
            print(col)
            tmp.loc[~tmp[col].isnull(), col] /= (
                tmp.loc[~tmp[col].isnull(), :].groupby(["date", "port"]).doi.transform("sum")
            )

        tmp = tmp.groupby(["date", "port"]).sum()  # numerator
        tmp = tmp.drop(columns=["doi", "rel_optspread", "non_exp_month"])
        # ---- weighting end ----

        cols = list(set([c.split("_")[0] for c in tmp.columns]))
        spreads = []
        for col in cols:
            spread = tmp[[c for c in tmp.columns if col in c]]
            short = spread[[c for c in spread.columns if "short" in c]]
            short = short[short.index.get_level_values("port") == 0].reset_index(level="port", drop=True)
            short.columns = [col]
            long = spread[[c for c in spread.columns if "short" not in c]]
            long = long[long.index.get_level_values("port") == 9].reset_index(level="port", drop=True)
            long.columns = [col]
            spreads.append(long - short)
            if col in model_returns:
                tt = model_returns[col]
                tt[model] = long - short
            else:
                model_returns[col] = {model: long - short}
        spreads = pd.concat(spreads, axis=1)
        spreads *= 100
        mean_ret = spreads.mean()
        mean_ret.name = int(level_spread * 100)
        out_df.append(mean_ret)

        t_values = []
        for col in spreads.columns:
            tmp = spreads[col]
            tmp = sm.OLS(tmp, np.ones(tmp.shape[0])).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues
            tmp.index = [col]
            t_values.append(tmp)
        t_values = pd.concat(t_values)
        t_values.name = int(level_spread * 100)
        out_df_tvals.append(t_values)

        sr_values = []
        for col in spreads.columns:
            tmp = spreads[col]
            tmp = tmp.mean() / tmp.std()
            tmp = pd.Series({col: tmp})
            sr_values.append(tmp)
        sr_values = pd.concat(sr_values)
        sr_values.name = int(level_spread * 100)
        out_df_sr.append(sr_values)

    out_df = pd.concat(out_df, axis=1)
    out_df = out_df.stack()
    out_df_tvals = pd.concat(out_df_tvals, axis=1)
    out_df_tvals = out_df_tvals.stack()
    out_df_sr = pd.concat(out_df_sr, axis=1)
    out_df_sr = out_df_sr.stack()
    out_df = pd.concat((out_df, out_df_tvals, out_df_sr), axis=1)
    out_df = out_df.reset_index()
    out_df.columns = ["spread", "level_spread", "ret", "tval", "sr"]
    out_df = out_df.drop(columns="level_spread")
    out_df = out_df.set_index("spread")
    out_df.columns = pd.MultiIndex.from_arrays([[model] * out_df.shape[1], out_df.columns])

    to_plot.append(out_df)

to_plot = pd.concat(to_plot, axis=1)
to_plot.insert(3, "EMPTY_0", "")
to_plot["EMPTY_1"] = ""
val = []
for idx, gp in to_plot.iterrows():
    ttest = ttest_ind(model_returns[idx]["N-En"], model_returns[idx]["L-En"])[0]
    val.append(ttest[0])
val = np.array(val)

to_plot[("", "N vs. L")] = ""
to_plot.loc[np.abs(val) > 1.65, ("", "N vs. L")] = "*"
to_plot.loc[np.abs(val) > 1.96, ("", "N vs. L")] = "**"
to_plot.loc[np.abs(val) > 2.56, ("", "N vs. L")] = "***"

to_plot_oprices = to_plot.copy()


# --- include delta-hedging costs

to_plot = []
model_returns = {}

for model in ["L-En", "N-En"]:
    print(model)
    data = prediction_dict[model]["predictions"].copy()
    data = data.set_index(["optionid"], append=True)
    data = data[["target", "predicted"]]
    data.columns = ["spread=0", "predicted"]
    data["spread=0_short"] = data["spread=0"]
    data = data.merge(spread_returns, on=["date", "optionid"])

    # ---- DOI-weighted portfolios, across underlyings
    data = data.merge(
        classifier[
            ["date", "optionid", "doi", "rel_optspread", "rel_S_spreads_paid", "exp_S_spreads_paid", "non_exp_month"]
        ],
        on=["date", "optionid"],
    )
    data = data.drop_duplicates()  # to be safe.

    out_df = []
    out_df_sr = []
    out_df_tvals = []
    for level_spread in levels_spreads:
        print(level_spread)
        # twice the spread as expected costs if option does not mature in considered month; otherwise just once the spread as E[costs]
        spread_costs = (
            data["rel_optspread"] * level_spread * (1 + data["non_exp_month"])
            + data["exp_S_spreads_paid"].abs() * level_spread
        ).to_numpy()
        tmp = data.copy()
        tmp.loc[tmp.predicted >= 0, "predicted"] = np.maximum(
            0, tmp.loc[tmp.predicted >= 0, "predicted"] - spread_costs[tmp.predicted >= 0]
        )
        tmp.loc[tmp.predicted < 0, "predicted"] = np.minimum(
            0, tmp.loc[tmp.predicted < 0, "predicted"] + spread_costs[tmp.predicted < 0]
        )

        # test
        tmp = tmp.loc[tmp.predicted != 0, :]
        # test end

        tmp["port"] = tmp.groupby("date").predicted.transform(
            lambda x: pd.qcut(x, n_ports, labels=False, duplicates="drop")
        )
        tmp = tmp[tmp["port"].isin([0, 9])]
        tmp = tmp.set_index(["date", "optionid"])
        tmp = tmp.drop(columns=["predicted"])
        tmp = tmp.drop(
            columns=[c for c in tmp.columns if "=" + str(int(level_spread * 100)) not in c and c.startswith("spread")]
        )

        long_col = [c for c in tmp.columns if (c.startswith("spread") and (not ("short" in c)))]
        for l_col in long_col:
            tmp[l_col] = tmp[l_col] - tmp.rel_S_spreads_paid.abs() * level_spread
        short_col = [c for c in tmp.columns if (c.startswith("spread") and ("short" in c))]
        for s_col in short_col:
            tmp[s_col] = tmp[s_col] + tmp.rel_S_spreads_paid.abs() * level_spread
        tmp = tmp.drop(columns=["rel_S_spreads_paid", "rel_optspread", "non_exp_month", "exp_S_spreads_paid"])

        weighted = tmp[[c for c in tmp.columns if c.startswith("spread")]].multiply(tmp["doi"], axis=0)
        tmp[weighted.columns] = weighted

        # needed if we truncated, since some DOI are "missing" if the return is "missing"
        for col in [c for c in tmp.columns if c.startswith("spread")]:
            print(col)
            tmp.loc[~tmp[col].isnull(), col] /= (
                tmp.loc[~tmp[col].isnull(), :].groupby(["date", "port"]).doi.transform("sum")
            )

        tmp = tmp.groupby(["date", "port"]).sum()  # numerator
        tmp
        tmp = tmp.drop(columns=["doi"])
        # ---- weighting end ----

        cols = list(set([c.split("_")[0] for c in tmp.columns]))
        spreads = []
        for col in cols:
            spread = tmp[[c for c in tmp.columns if col in c]]
            short = spread[[c for c in spread.columns if "short" in c]]
            short = short[short.index.get_level_values("port") == 0].reset_index(level="port", drop=True)
            short.columns = [col]
            long = spread[[c for c in spread.columns if "short" not in c]]
            long = long[long.index.get_level_values("port") == 9].reset_index(level="port", drop=True)
            long.columns = [col]
            spreads.append(long - short)
            if col in model_returns:
                tt = model_returns[col]
                tt[model] = long - short
            else:
                model_returns[col] = {model: long - short}
        spreads = pd.concat(spreads, axis=1)
        spreads *= 100
        mean_ret = spreads.mean()
        mean_ret.name = int(level_spread * 100)
        out_df.append(mean_ret)

        t_values = []
        for col in spreads.columns:
            tmp = spreads[col]
            tmp = sm.OLS(tmp, np.ones(tmp.shape[0])).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues
            tmp.index = [col]
            t_values.append(tmp)
        t_values = pd.concat(t_values)
        t_values.name = int(level_spread * 100)
        out_df_tvals.append(t_values)

        sr_values = []
        for col in spreads.columns:
            tmp = spreads[col]
            tmp = tmp.mean() / tmp.std()
            tmp = pd.Series({col: tmp})
            sr_values.append(tmp)
        sr_values = pd.concat(sr_values)
        sr_values.name = int(level_spread * 100)
        out_df_sr.append(sr_values)

    out_df = pd.concat(out_df, axis=1)
    out_df = out_df.stack()
    out_df_tvals = pd.concat(out_df_tvals, axis=1)
    out_df_tvals = out_df_tvals.stack()
    out_df_sr = pd.concat(out_df_sr, axis=1)
    out_df_sr = out_df_sr.stack()
    out_df = pd.concat((out_df, out_df_tvals, out_df_sr), axis=1)
    out_df = out_df.reset_index()
    out_df.columns = ["spread", "level_spread", "ret", "tval", "sr"]
    out_df = out_df.drop(columns="level_spread")
    out_df = out_df.set_index("spread")
    out_df.columns = pd.MultiIndex.from_arrays([[model] * out_df.shape[1], out_df.columns])

    to_plot.append(out_df)

to_plot = pd.concat(to_plot, axis=1)
to_plot.insert(3, "EMPTY_0", "")
to_plot["EMPTY_1"] = ""
val = []
for idx, gp in to_plot.iterrows():
    ttest = ttest_ind(model_returns[idx]["N-En"], model_returns[idx]["L-En"])[0]
    val.append(ttest[0])
val = np.array(val)

to_plot[("", "N vs. L")] = ""
to_plot.loc[np.abs(val) > 1.65, ("", "N vs. L")] = "*"
to_plot.loc[np.abs(val) > 1.96, ("", "N vs. L")] = "**"
to_plot.loc[np.abs(val) > 2.56, ("", "N vs. L")] = "***"

to_plot_oprices_and_dh = to_plot.copy()


# --- include margin requirements

levels_spreads = [0.25, 0.5, 0.75, 1]

to_plot = []
model_returns = {}

for model in ["L-En", "N-En"]:
    print(model)
    data = prediction_dict[model]["predictions"].copy()
    data = data.set_index(["optionid"], append=True)
    data = data[["target", "predicted"]]
    data.columns = ["spread=0", "predicted"]
    data["spread=0_short"] = data["spread=0"]
    data = data.merge(spread_returns, on=["date", "optionid"])

    # ---- DOI-weighted portfolios, across underlyings
    data = data.merge(
        classifier[
            [
                "date",
                "optionid",
                "doi",
                "denominator",
                "margin_denominator",
                "margin_denominator_signed",
                "rel_optspread",
                "rel_S_spreads_paid",
                "exp_S_spreads_paid",
                "non_exp_month",
            ]
        ],
        on=["date", "optionid"],
    )
    data = data.drop_duplicates()  # to be safe.

    out_df = []
    out_df_sr = []
    out_df_tvals = []
    for level_spread in levels_spreads:
        print(level_spread)
        # twice the spread as expected costs if option does not mature in considered month; otherwise just once the spread as E[costs]
        # + transform predicted returns from /denom to /margin_denom
        spread_costs = (
            data["rel_optspread"] * level_spread * (1 + data["non_exp_month"])
            + data["exp_S_spreads_paid"].abs() * level_spread
        ).to_numpy()
        tmp = data.copy()
        tmp.loc[tmp.predicted >= 0, "predicted"] = np.maximum(
            0,
            (tmp.loc[tmp.predicted >= 0, "predicted"] - spread_costs[tmp.predicted >= 0])
            * tmp["denominator"]
            / tmp["margin_denominator"],
        )
        tmp.loc[tmp.predicted < 0, "predicted"] = np.minimum(
            0,
            (tmp.loc[tmp.predicted < 0, "predicted"] + spread_costs[tmp.predicted < 0])
            * tmp["denominator"]
            / tmp["margin_denominator_signed"],
        )

        # test
        tmp = tmp.loc[tmp.predicted != 0, :]
        # test end

        # then sort into portfolios
        tmp["port"] = tmp.groupby("date").predicted.transform(
            lambda x: pd.qcut(x, n_ports, labels=False, duplicates="drop")
        )
        tmp = tmp[tmp["port"].isin([0, 9])]
        tmp = tmp.set_index(["date", "optionid"])
        tmp = tmp.drop(columns=["predicted"])
        tmp = tmp.drop(
            columns=[c for c in tmp.columns if str(int(level_spread * 100)) not in c and c.startswith("spread")]
        )

        # transform realized returns from /denom to /margin_denom
        long_col = [c for c in tmp.columns if (c.startswith("spread") and (not ("short" in c)))]
        for l_col in long_col:
            tmp[l_col] = tmp[l_col] - tmp.rel_S_spreads_paid.abs() * level_spread
            tmp[l_col] = tmp[l_col] * tmp.denominator / tmp.margin_denominator.abs()
        short_col = [c for c in tmp.columns if (c.startswith("spread") and ("short" in c))]
        for s_col in short_col:
            tmp[s_col] = tmp[s_col] + tmp.rel_S_spreads_paid.abs() * level_spread
            tmp[s_col] = tmp[s_col] * tmp.denominator / tmp.margin_denominator_signed.abs()

        tmp = tmp.drop(
            columns=[
                "denominator",
                "margin_denominator",
                "margin_denominator_signed",
                "rel_optspread",
                "rel_S_spreads_paid",
                "exp_S_spreads_paid",
                "non_exp_month",
            ]
        )

        weighted = tmp[[c for c in tmp.columns if c.startswith("spread")]].multiply(tmp["doi"], axis=0)
        tmp[weighted.columns] = weighted

        # needed if we truncated, since some DOI are "missing" if the return is "missing"
        for col in [c for c in tmp.columns if c.startswith("spread")]:
            print(col)
            tmp.loc[~tmp[col].isnull(), col] /= (
                tmp.loc[~tmp[col].isnull(), :].groupby(["date", "port"]).doi.transform("sum")
            )

        tmp = tmp.groupby(["date", "port"]).sum()  # numerator

        tmp = tmp.drop(columns=["doi"])
        # ---- weighting end ----

        cols = list(set([c.split("_")[0] for c in tmp.columns]))
        spreads = []
        for col in cols:
            spread = tmp[[c for c in tmp.columns if col in c]]
            short = spread[[c for c in spread.columns if "short" in c]]
            short = short[short.index.get_level_values("port") == 0].reset_index(level="port", drop=True)
            short.columns = [col]
            long = spread[[c for c in spread.columns if "short" not in c]]
            long = long[long.index.get_level_values("port") == 9].reset_index(level="port", drop=True)
            long.columns = [col]
            spreads.append(long - short)
            if col in model_returns:
                tt = model_returns[col]
                tt[model] = long - short
            else:
                model_returns[col] = {model: long - short}
        spreads = pd.concat(spreads, axis=1)
        spreads *= 100
        mean_ret = spreads.mean()
        mean_ret.name = int(level_spread * 100)
        out_df.append(mean_ret)

        t_values = []
        for col in spreads.columns:
            tmp = spreads[col]
            tmp = sm.OLS(tmp, np.ones(tmp.shape[0])).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues
            tmp.index = [col]
            t_values.append(tmp)
        t_values = pd.concat(t_values)
        t_values.name = int(level_spread * 100)
        out_df_tvals.append(t_values)

        sr_values = []
        for col in spreads.columns:
            tmp = spreads[col]
            tmp = tmp.mean() / tmp.std()
            tmp = pd.Series({col: tmp})
            sr_values.append(tmp)
        sr_values = pd.concat(sr_values)
        sr_values.name = int(level_spread * 100)
        out_df_sr.append(sr_values)

    out_df = pd.concat(out_df, axis=1)
    out_df = out_df.stack()
    out_df_tvals = pd.concat(out_df_tvals, axis=1)
    out_df_tvals = out_df_tvals.stack()
    out_df_sr = pd.concat(out_df_sr, axis=1)
    out_df_sr = out_df_sr.stack()
    out_df = pd.concat((out_df, out_df_tvals, out_df_sr), axis=1)
    out_df = out_df.reset_index()
    out_df.columns = ["spread", "level_spread", "ret", "tval", "sr"]
    out_df = out_df.drop(columns="level_spread")
    out_df = out_df.set_index("spread")
    out_df.columns = pd.MultiIndex.from_arrays([[model] * out_df.shape[1], out_df.columns])

    to_plot.append(out_df)

to_plot = pd.concat(to_plot, axis=1)
to_plot.insert(3, "EMPTY_0", "")
to_plot["EMPTY_1"] = ""
val = []
for idx, gp in to_plot.iterrows():
    ttest = ttest_ind(model_returns[idx]["N-En"], model_returns[idx]["L-En"])[0]
    val.append(ttest[0])
val = np.array(val)

to_plot[("", "N vs. L")] = ""
to_plot.loc[np.abs(val) > 1.65, ("", "N vs. L")] = "*"
to_plot.loc[np.abs(val) > 1.96, ("", "N vs. L")] = "**"
to_plot.loc[np.abs(val) > 2.56, ("", "N vs. L")] = "***"

to_plot_oprices_and_dh_margin = to_plot.copy()


def latex_table_tc(data, data_names, save_name: str, num_format="%.4f", tval_format="%.2f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    def col_renamer(c):
        renamer = {"ret": "H-L", "tval": "t", "sr": "SR", "N vs. L": ""}
        if c in renamer.keys():
            return renamer[c]
        else:
            return c

    table = Tabular("".join(["l"] + ["c"] * (data[0].shape[1] - 1) + ["c"]), booktabs=True)
    table.add_row(
        [
            "Eff. Spread",
            MultiColumn(3, align="c", data=data[0].columns.get_level_values(0)[0]),
            "",
            MultiColumn(3, align="c", data=data[0].columns.get_level_values(0)[-3]),
            "",
            "N vs. L",
        ]
    )
    table.add_hline(start=2, end=4)
    table.add_hline(start=6, end=8)
    # table.add_hline(start=10, end=10)
    to_add = [""] + ["" if "EMPTY" in c else col_renamer(c) for c in data[0].columns.get_level_values(1).tolist()]
    table.add_row([MultiColumn(1, align="c", data=c) for c in to_add])
    table.add_hline()

    for i in range(0, len(data)):
        ttmp = data[i]
        pname = data_names[i]
        if pname != "":
            table.add_empty_row()
            table.add_row([MultiColumn(ttmp.shape[1] + 1, align="c", data=pname)])
            table.add_hline()

        for i, (idx, row) in enumerate(ttmp.iterrows()):
            to_add = []
            for col, num in row.iteritems():
                if isinstance(num, float):
                    if np.isnan(num):
                        to_add.append("")
                    else:
                        if col[1] == "tval":
                            num = "%.2f" % num
                            to_add.append(math(r"\footnotesize{(%s)}" % num))
                        else:
                            num = num_format % num
                            to_add.append(math(num))
                else:
                    to_add.append(num)
            s = idx.split("=")[1]
            table.add_row([math(r"s\%".replace("s", s))] + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


latex_table_tc(
    [
        to_plot_oprices,
        to_plot_oprices_and_dh,
        to_plot_oprices_and_dh_margin,
    ],
    [
        "Option Costs",
        "Option And Delta-Hedging Costs",
        "Option And Delta-Hedging Costs with Long/Short Margin Requirements",
    ],
    save_name="trading_strat_spreads",
    num_format="%.3f",
)
