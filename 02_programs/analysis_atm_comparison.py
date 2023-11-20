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
from scoring import TS_R2, DieboldMariano, DieboldMariano_XS, XS_R2

import matplotlib.pyplot as plt

from joblib import load

# import matplotlib
import seaborn as sns

from analysis_setup import prediction_dict, sns_palette, width, height, analysisLoc, modelLoc

class_groups = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups.pq"), columns=["date", "optionid", "type", "bucket"]
)

atm_dict = load(os.path.join(modelLoc, "prediction_dict_atm.pkl"))


# %%
# ATM model vs. full model for *ALL* options: R2
for i, model in enumerate(prediction_dict):
    print(model)
    tmp = prediction_dict[model]["predictions"].copy().reset_index()
    tmp = tmp.rename(columns={"predicted": model})
    tmp = tmp.sort_values(["date", "optionid"])
    if i == 0:
        to_plot = tmp[["date", "optionid", "target", model]]
    else:
        to_plot = pd.concat((to_plot, tmp[model]), axis=1)

to_plot = to_plot.drop(columns=["optionid"])
to_plot = to_plot.set_index("date")
r2 = TS_R2(to_plot)
r2 = r2.to_frame()
r2.columns = ["score"]
r2["Estimator"] = "Full"

atm = []
for i, model in enumerate(prediction_dict):
    if i == 0:
        target = atm_dict[model]["predictions"]["target"].copy()
    tmp = atm_dict[model]["predictions"]["predicted"]
    tmp.name = model
    atm.append(tmp.copy())
atm = pd.concat(atm, axis=1)
atm = pd.concat((atm, target), axis=1)
r2_atm = TS_R2(atm)
r2_atm = r2_atm.to_frame()
r2_atm.columns = ["score"]
r2_atm["Estimator"] = "ATM"

r2 = pd.concat((r2, r2_atm))
r2 = r2.reset_index()
r2.columns = ["model", "score", "Estimator"]


fig, axes = plt.subplots(figsize=(width, height * 0.6), dpi=1000)
sns.barplot(x="model", y="score", hue="Estimator", data=r2, ax=axes, palette=sns_palette(2))
axes.set_ylabel("")
axes.set_xlabel("")
axes.legend(frameon=False, title=None)
axes.axhline(0, color="k", ls="--")
fig.tight_layout()
axes.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
axes.set_axisbelow(True)
fig.savefig("../08_figures/atm_vs_full_r2.pdf", bbox_inches="tight")


# %%
# ATM model vs. full model for *ALL* options: R2 XS
to_plot = []
for i, model in enumerate(prediction_dict):
    if i == 0:
        target = prediction_dict[model]["predictions"]["target"].copy()
    tmp = prediction_dict[model]["predictions"]["predicted"]
    tmp.name = model
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot = pd.concat((to_plot, target), axis=1)
r2 = XS_R2(to_plot)
r2 = r2.to_frame()
r2.columns = ["score"]
r2["Estimator"] = "Full"

atm = []
for i, model in enumerate(prediction_dict):
    if i == 0:
        target = atm_dict[model]["predictions"]["target"].copy()
    tmp = atm_dict[model]["predictions"]["predicted"]
    tmp.name = model
    atm.append(tmp.copy())
atm = pd.concat(atm, axis=1)
atm = pd.concat((atm, target), axis=1)
r2_atm = XS_R2(atm)
r2_atm = r2_atm.to_frame()
r2_atm.columns = ["score"]
r2_atm["Estimator"] = "ATM"

r2 = pd.concat((r2, r2_atm))
r2 = r2.reset_index()
r2.columns = ["model", "score", "Estimator"]


fig, axes = plt.subplots(figsize=(width, height * 0.6), dpi=1000)
sns.barplot(x="model", y="score", hue="Estimator", data=r2, ax=axes, palette=sns_palette(2))
axes.set_ylabel("")
axes.set_xlabel("")
axes.legend(frameon=False, title=None)
axes.axhline(0, color="k", ls="--")
fig.tight_layout()
axes.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
axes.set_axisbelow(True)
fig.savefig("../08_figures/atm_vs_full_r2_xs.pdf", bbox_inches="tight")


# %%
# ATM model vs. full model for *ATM ONLY*: R2
to_plot = []
for i, model in enumerate(prediction_dict):
    print(model)
    if i == 0:
        target = prediction_dict[model]["predictions"][["optionid", "target"]].copy()
        target = target.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])
        target = target[target.bucket == "short_term_atm"]
        target = target.sort_values(["date", "optionid"])
        target = target.drop(columns=["bucket"])
        target = target.set_index("date")
    tmp = prediction_dict[model]["predictions"][["optionid", "predicted"]].copy()
    tmp = tmp.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])
    tmp = tmp[tmp.bucket == "short_term_atm"]
    tmp = tmp.sort_values(["date", "optionid"])
    tmp = tmp.drop(columns=["bucket", "optionid"])
    tmp = tmp.set_index("date")
    tmp.columns = [model]
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot = pd.concat((to_plot, target), axis=1)
r2 = TS_R2(to_plot.drop(columns=["optionid"]))
r2 = r2.to_frame()
r2.columns = ["score"]
r2["Estimator"] = "Full"


atm = []
for i, model in enumerate(prediction_dict):
    print(model)
    if i == 0:
        target = atm_dict[model]["predictions"][["optionid", "target"]].copy()
        target = target.reset_index()
        target = target.sort_values(["date", "optionid"])
        target = target.set_index("date")
    tmp = atm_dict[model]["predictions"][["optionid", "predicted"]].copy()
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(["date", "optionid"])
    tmp = tmp.drop(columns=["optionid"])
    tmp = tmp.set_index("date")
    tmp.columns = [model]
    atm.append(tmp.copy())
atm = pd.concat(atm, axis=1)
atm = pd.concat((atm, target), axis=1)
r2_atm = TS_R2(atm.drop(columns=["optionid"]))
r2_atm = r2_atm.to_frame()
r2_atm.columns = ["score"]
r2_atm["Estimator"] = "ATM"

r2 = pd.concat((r2, r2_atm))
r2 = r2.reset_index()
r2.columns = ["model", "score", "Estimator"]


# DM test for differences between the two:
dm_test = []
for i, model in enumerate(prediction_dict):
    print(model)
    tmp = to_plot[["optionid", model]]
    tmp.columns = ["optionid", "full"]
    tmp["target"] = to_plot["target"]
    tmp = tmp.set_index("optionid", append=True)
    tmp = tmp.merge(atm.set_index("optionid", append=True)[[model]], on=["date", "optionid"], how="inner")
    tmp = tmp.rename(columns={model: "atm"})
    diebold_test = DieboldMariano(tmp, 12)
    dm_test.append(diebold_test.values[0][0])

stars = []
for c in dm_test:
    if c > 3.09:
        stars.append("***")
    elif c > 2.326:
        stars.append("**")
    elif c > 1.645:
        stars.append("*")
    else:
        stars.append("")
stars = pd.Series(data=stars)
# ----

fig, axes = plt.subplots(figsize=(width, height * 0.6), dpi=1000)
sns.barplot(x="model", y="score", hue="Estimator", data=r2, ax=axes, palette=sns_palette(2))
axes.set_ylabel("")
axes.set_xlabel("")
axes.legend(frameon=False, title=None)
axes.axhline(0, color="k", ls="--")
fig.tight_layout()
axes.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
axes.set_axisbelow(True)
for i, (_, txt) in enumerate(stars.iteritems()):
    axes.annotate(txt, (i, -0.0295), ha="center")
fig.savefig("../08_figures/atm_vs_full_r2_ONLY_ATM.pdf", bbox_inches="tight")


# %%
# ATM model vs. full model for *ATM ONLY*: R2 XS
to_plot = []
for i, model in enumerate(prediction_dict):
    print(model)
    if i == 0:
        target = prediction_dict[model]["predictions"][["optionid", "target"]].copy()
        target = target.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])
        target = target[target.bucket == "short_term_atm"]
        target = target.sort_values(["date", "optionid"])
        target = target.drop(columns=["bucket"])
        target = target.set_index("date")
    tmp = prediction_dict[model]["predictions"][["optionid", "predicted"]].copy()
    tmp = tmp.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])
    tmp = tmp[tmp.bucket == "short_term_atm"]
    tmp = tmp.sort_values(["date", "optionid"])
    tmp = tmp.drop(columns=["bucket", "optionid"])
    tmp = tmp.set_index("date")
    tmp.columns = [model]
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot = pd.concat((to_plot, target), axis=1)
r2 = XS_R2(to_plot.drop(columns=["optionid"]))
r2 = r2.to_frame()
r2.columns = ["score"]
r2["Estimator"] = "Full"


atm = []
for i, model in enumerate(prediction_dict):
    print(model)
    if i == 0:
        target = atm_dict[model]["predictions"][["optionid", "target"]].copy()
        target = target.reset_index()
        target = target.sort_values(["date", "optionid"])
        target = target.set_index("date")
    tmp = atm_dict[model]["predictions"][["optionid", "predicted"]].copy()
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(["date", "optionid"])
    tmp = tmp.drop(columns=["optionid"])
    tmp = tmp.set_index("date")
    tmp.columns = [model]
    atm.append(tmp.copy())
atm = pd.concat(atm, axis=1)
atm = pd.concat((atm, target), axis=1)
r2_atm = XS_R2(atm.drop(columns=["optionid"]))
r2_atm = r2_atm.to_frame()
r2_atm.columns = ["score"]
r2_atm["Estimator"] = "ATM"

r2 = pd.concat((r2, r2_atm))
r2 = r2.reset_index()
r2.columns = ["model", "score", "Estimator"]


# DM test for differences between the two:
dm_test = []
for i, model in enumerate(prediction_dict):
    print(model)
    tmp = to_plot[["optionid", model]]
    tmp.columns = ["optionid", "full"]
    tmp["target"] = to_plot["target"]
    tmp = tmp.set_index("optionid", append=True)
    tmp = tmp.merge(atm.set_index("optionid", append=True)[[model]], on=["date", "optionid"], how="inner")
    tmp = tmp.rename(columns={model: "atm"})
    diebold_test = DieboldMariano_XS(tmp, 12)
    dm_test.append(diebold_test.values[0][0])

stars = []
for c in dm_test:
    if c > 3.09:
        stars.append("***")
    elif c > 2.326:
        stars.append("**")
    elif c > 1.645:
        stars.append("*")
    else:
        stars.append("")
stars = pd.Series(data=stars)
# ----

fig, axes = plt.subplots(figsize=(width, height * 0.6), dpi=1000)
sns.barplot(x="model", y="score", hue="Estimator", data=r2, ax=axes, palette=sns_palette(2))
axes.set_ylabel("")
axes.set_xlabel("")
axes.legend(frameon=False, title=None)
axes.axhline(0, color="k", ls="--")
fig.tight_layout()
axes.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
axes.set_axisbelow(True)
for i, (_, txt) in enumerate(stars.iteritems()):
    axes.annotate(txt, (i, -0.014), ha="center")
fig.savefig("../08_figures/atm_vs_full_r2_xs_ONLY_ATM.pdf", bbox_inches="tight")


# %% Functions
from numba import njit
import statsmodels.api as sm
from scipy.stats import ttest_ind


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


# %% Trading strategy for the full sample
hml_returns = {}
to_plot = []
model_returns = {}
for model in ["L-En", "N-En"]:
    tmp = atm_dict[model]["predictions"].copy()
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
