# %%
""" Comparative analyses for different types of return calculations or subsamples. """


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

import glob

# import matplotlib
import seaborn as sns

from analysis_setup import sns_palette, width, height, analysisLoc, modelLoc

class_groups = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups.pq"), columns=["date", "optionid", "type", "bucket"]
)
class_groups_with_info = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups_and_info.pq"),
    columns=["date", "optionid", "doi", "leverage"],
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
# MARGIN RETURNS
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
add_on = "margin"
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

# %%
# Trading strategy TRANSACTION COSTS using **value-weights*:

n_ports = 10
spread_returns = pd.read_parquet("../03_data/returns_spreads.pq")
spread_returns = spread_returns.reset_index()
spread_returns["optionid"] = spread_returns["optionid"].astype("int")
spread_returns["date"] = spread_returns["date"].dt.to_period("M")
spread_returns = spread_returns.set_index(["date", "optionid"])

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
classifier["rel_optspread"] = classifier["optspread"] / 2 / classifier["margin_denominator"]
classifier["rel_S_spreads_paid"] = classifier.S_spreads_paid / 2 / classifier["margin_denominator"]
classifier["exp_S_spreads_paid"] = (
    classifier.baspread / 2 * 3 * classifier.delta.abs() / classifier["margin_denominator"]
)

# Game spreads by considering not just E[r] but E[r] minus spreads
levels_spreads = [0, 0.25, 0.5, 0.75, 1]

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
    data = data.merge(spread_returns, on=["date", "optionid"])

    # ---- DOI-weighted portfolios, across underlyings
    data = data.merge(
        classifier[
            [
                "date",
                "optionid",
                "doi",
                "denominator",
                "rel_optspread",
                "non_exp_month",
                "margin_denominator",
                "margin_denominator_signed",
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
        spread_costs = (data["rel_optspread"] * level_spread * (1 + data["non_exp_month"])).to_numpy()
        tmp = data.copy()
        tmp.loc[tmp.predicted >= 0, "predicted"] = np.maximum(
            0, tmp.loc[tmp.predicted >= 0, "predicted"] - spread_costs[tmp.predicted >= 0]
        )
        tmp.loc[tmp.predicted < 0, "predicted"] = np.minimum(
            0, tmp.loc[tmp.predicted < 0, "predicted"] + spread_costs[tmp.predicted < 0]
        )

        tmp = tmp.loc[tmp.predicted != 0, :]

        tmp["port"] = tmp.groupby("date").predicted.transform(
            lambda x: pd.qcut(x, n_ports, labels=False, duplicates="drop")
        )
        tmp = tmp[tmp["port"].isin([0, 9])]
        tmp = tmp.set_index(["date", "optionid"])
        tmp = tmp.drop(columns=["predicted"])
        tmp = tmp.drop(
            columns=[c for c in tmp.columns if "=" + str(int(level_spread * 100)) not in c and c.startswith("spread")]
        )

        # check how returns are calculated: margin adjusted or not
        short_col = [c for c in tmp.columns if c.startswith("spread") and ("short" in c)]
        for s_col in short_col:
            if level_spread > 0:
                tmp[s_col] = tmp[s_col] * tmp.denominator / tmp.margin_denominator_signed.abs()
            else:
                tmp[s_col] = tmp[s_col] * tmp.margin_denominator / tmp.margin_denominator_signed.abs()
        long_col = [c for c in tmp.columns if c.startswith("spread") and ("short" not in c)]
        for l_col in long_col:
            if level_spread > 0:
                tmp[l_col] = tmp[l_col] * tmp.denominator / tmp.margin_denominator.abs()

        tmp = tmp.drop(columns=["denominator", "margin_denominator", "margin_denominator_signed"])

        weighted = tmp[[c for c in tmp.columns if c.startswith("spread")]].multiply(tmp["doi"], axis=0)
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

# include delta-hedging costs
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
                "rel_optspread",
                "rel_S_spreads_paid",
                "non_exp_month",
                "exp_S_spreads_paid",
                "denominator",
                "margin_denominator",
                "margin_denominator_signed",
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
            columns=[c for c in tmp.columns if str(int(level_spread * 100)) not in c and c.startswith("spread")]
        )

        long_col = [c for c in tmp.columns if (c.startswith("spread") and (not ("short" in c)))]
        for l_col in long_col:
            if level_spread > 0:
                tmp[l_col] = tmp[l_col] * tmp.denominator / tmp.margin_denominator.abs()
            tmp[l_col] = tmp[l_col] - tmp.rel_S_spreads_paid.abs() * level_spread
        short_col = [c for c in tmp.columns if (c.startswith("spread") and ("short" in c))]
        for s_col in short_col:
            if level_spread > 0:
                tmp[s_col] = tmp[s_col] * tmp.denominator / tmp.margin_denominator.abs()
            tmp[s_col] = tmp[s_col] + tmp.rel_S_spreads_paid.abs() * level_spread
            tmp[s_col] = tmp[s_col] * tmp.margin_denominator.abs() / tmp.margin_denominator_signed.abs()

        tmp = tmp.drop(
            columns=[
                "rel_optspread",
                "non_exp_month",
                "rel_S_spreads_paid",
                "exp_S_spreads_paid",
                "denominator",
                "margin_denominator",
                "margin_denominator_signed",
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

to_plot_oprices_and_dh = to_plot.copy()


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
            table.add_row(["", MultiColumn(ttmp.shape[1] - 1, align="c", data=pname), ""])
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
    [pd.DataFrame(to_plot_oprices.iloc[0]).T, to_plot_oprices.iloc[1:], to_plot_oprices_and_dh],
    ["No Transaction Costs", "Option Costs", "Option And Delta-Hedging Costs"],
    save_name="trading_strat_spreads_signed_margin",
    num_format="%.3f",
)

# %%
# DELEVERED RETURNS JACKWERTH
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
add_on = "delevered_jackwerth"
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
# Addresses Weak vs. Strong factors to some Degree.
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
to_plot.index = to_plot.index.to_period("M")
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


fig, axes = plt.subplots(1, 2, figsize=(width, height), dpi=1000, sharey=True)
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
    tmp.index = tmp.index.to_period("M")

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
    tmp.index = tmp.index.to_period("M")
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


# %%
# DELEVERED RETURNS
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
add_on = "delevered"
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
# Addresses Weak vs. Strong factors to some Degree.
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
to_plot.index = to_plot.index.to_period("M")
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


fig, axes = plt.subplots(1, 2, figsize=(width, height), dpi=1000, sharey=True)
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
    tmp.index = tmp.index.to_period("M")

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
    tmp.index = tmp.index.to_period("M")
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


# %%
# DELEVERED USING FULL MODEL
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
add_on = "delevered_full_model"
prediction_dict = load(os.path.join(modelLoc, "prediction_dict_comparison.pkl"))


# %%
# Trading strategy using value weights
# Shows Pred/Avg/SD/Sharpe/Average Leverage/Avg delevered using avg leverage/ Avg delevered using leverage by time/ Avg deleverd using leverage by optionXtime
to_plot = []
model_returns = {}

# ----------- A) divide by full-sample average of leverage per portfolio
model = r"\text{Lev}_p"
tmp = prediction_dict["N-En"]["predictions"].copy()
# ---- DOI-weighted portfolios, across underlyings
tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi", "leverage"]], on=["date", "optionid"])
weighted = tmp[["predicted", "target", "leverage"]].multiply(tmp["doi"], axis=0)
tmp[weighted.columns] = weighted
tmp = tmp.drop(columns=["optionid"])

total_weights = tmp.groupby(["date", "port"]).doi.sum()  # denominator
total_weights.name = "summed_weights"
tmp = tmp.groupby(["date", "port"]).sum()  # numerator
tmp = tmp.divide(total_weights, axis=0)
tmp = tmp.drop(columns=["doi"])

tmp[["target", "predicted"]] = tmp[["target", "predicted"]].divide(
    tmp["leverage"].groupby("port").transform("mean"), axis=0
)

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
        lambda x: sm.OLS(x["target"], np.ones(x["target"].shape)).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues
    ).loc["H-L", "const"],
    np.nan,
    Lo_tval_SR(output[output.index.get_level_values("port") == "H-L"]["target"]),
]
strat.loc["tval"] = tvals
strat = strat[[(model, "Avg"), (model, "SR")]]
to_plot.append(strat)


# ------------- B) divide by time-t specific leverage per portfolio
model = r"\text{Lev}_{t,p}"
tmp = prediction_dict["N-En"]["predictions"].copy()
# ---- DOI-weighted portfolios, across underlyings
tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi", "leverage"]], on=["date", "optionid"])
weighted = tmp[["predicted", "target", "leverage"]].multiply(tmp["doi"], axis=0)
tmp[weighted.columns] = weighted
tmp = tmp.drop(columns=["optionid"])

total_weights = tmp.groupby(["date", "port"]).doi.sum()  # denominator
total_weights.name = "summed_weights"
tmp = tmp.groupby(["date", "port"]).sum()  # numerator
tmp = tmp.divide(total_weights, axis=0)
tmp = tmp.drop(columns=["doi"])

tmp[["target", "predicted"]] = tmp[["target", "predicted"]].divide(tmp["leverage"], axis=0)

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
        lambda x: sm.OLS(x["target"], np.ones(x["target"].shape)).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues
    ).loc["H-L", "const"],
    np.nan,
    Lo_tval_SR(output[output.index.get_level_values("port") == "H-L"]["target"]),
]
strat.loc["tval"] = tvals
strat = strat[[(model, "Avg"), (model, "SR")]]
to_plot.append(strat)


# ------------- C) divide by option X time-specific leverage
model = r"\text{Lev}_{t,o}"
tmp = prediction_dict["N-En"]["predictions"].copy()
# ---- DOI-weighted portfolios, across underlyings
tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi", "leverage"]], on=["date", "optionid"])
tmp[["target", "predicted"]] = tmp[["target", "predicted"]].divide(tmp["leverage"], axis=0)
weighted = tmp[["predicted", "target", "leverage"]].multiply(tmp["doi"], axis=0)
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
        lambda x: sm.OLS(x["target"], np.ones(x["target"].shape)).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues
    ).loc["H-L", "const"],
    np.nan,
    Lo_tval_SR(output[output.index.get_level_values("port") == "H-L"]["target"]),
]
strat.loc["tval"] = tvals
strat = strat[[(model, "Avg"), (model, "SR")]]
to_plot.append(strat)


# fill table:
to_plot = pd.concat(to_plot, axis=1)
to_plot.insert(2, "EMPTY_0", "")
to_plot.insert(4 + 1, "EMPTY_1", "")

full_sample = to_plot.copy()


# ---- split by puts and calls (only hml)
to_plot = []
model_returns = {}


# ----------- A) divide by full-sample average of leverage per portfolio
model = r"\text{Lev}_p"
col = "type"
tmp = prediction_dict["N-En"]["predictions"].copy()
tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])
tmp["port"] = tmp.groupby(["date", col]).predicted.transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
tmp = tmp.set_index("date")
# ---- DOI-weighted portfolios, across underlyings
tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi", "leverage"]], on=["date", "optionid"])
weighted = tmp[["predicted", "target", "leverage"]].multiply(tmp["doi"], axis=0)
tmp[weighted.columns] = weighted
tmp = tmp.drop(columns=["optionid"])

total_weights = tmp.groupby(["date", col, "port"]).doi.sum()  # denominator
total_weights.name = "summed_weights"
tmp = tmp.groupby(["date", col, "port"]).sum()  # numerator
tmp = tmp.divide(total_weights, axis=0)
tmp = tmp.drop(columns=["doi"])

tmp[["target", "predicted"]] = tmp[["target", "predicted"]].divide(
    tmp["leverage"].groupby([col, "port"]).transform("mean"), axis=0
)

output = []
for target_col in ["predicted", "target"]:
    returns = tmp[target_col].unstack()
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
strat = strat[[(model, "Avg"), (model, "SR")]]
to_plot.append(strat)


# ------------- B) divide by time-t specific leverage per portfolio
model = r"\text{Lev}_{t,p}"
col = "type"
tmp = prediction_dict["N-En"]["predictions"].copy()
tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])
tmp["port"] = tmp.groupby(["date", col]).predicted.transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
tmp = tmp.set_index("date")
# ---- DOI-weighted portfolios, across underlyings
tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi", "leverage"]], on=["date", "optionid"])
weighted = tmp[["predicted", "target", "leverage"]].multiply(tmp["doi"], axis=0)
tmp[weighted.columns] = weighted
tmp = tmp.drop(columns=["optionid"])

total_weights = tmp.groupby(["date", col, "port"]).doi.sum()  # denominator
total_weights.name = "summed_weights"
tmp = tmp.groupby(["date", col, "port"]).sum()  # numerator
tmp = tmp.divide(total_weights, axis=0)
tmp = tmp.drop(columns=["doi"])

tmp[["target", "predicted"]] = tmp[["target", "predicted"]].divide(tmp["leverage"], axis=0)

output = []
for target_col in ["predicted", "target"]:
    returns = tmp[target_col].unstack()
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
strat = strat[[(model, "Avg"), (model, "SR")]]
to_plot.append(strat)


# ------------- C) divide by option X time-specific leverage
model = r"\text{Lev}_{t,o}"
col = "type"
tmp = prediction_dict["N-En"]["predictions"].copy()
tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])
tmp["port"] = tmp.groupby(["date", col]).predicted.transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
tmp = tmp.set_index("date")
# ---- DOI-weighted portfolios, across underlyings
tmp = tmp.merge(class_groups_with_info[["date", "optionid", "doi", "leverage"]], on=["date", "optionid"])
tmp[["target", "predicted"]] = tmp[["target", "predicted"]].divide(tmp["leverage"], axis=0)
weighted = tmp[["predicted", "target", "leverage"]].multiply(tmp["doi"], axis=0)
tmp[weighted.columns] = weighted
tmp = tmp.drop(columns=["optionid"])

total_weights = tmp.groupby(["date", col, "port"]).doi.sum()  # denominator
total_weights.name = "summed_weights"
tmp = tmp.groupby(["date", col, "port"]).sum()  # numerator
tmp = tmp.divide(total_weights, axis=0)
tmp = tmp.drop(columns=["doi"])

output = []
for target_col in ["predicted", "target"]:
    returns = tmp[target_col].unstack()
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
strat = strat[[(model, "Avg"), (model, "SR")]]
to_plot.append(strat)


# fill table
to_plot = pd.concat(to_plot, axis=1)
to_plot.insert(2, "EMPTY_0", "")
to_plot.insert(4 + 1, "EMPTY_1", "")
to_plot = to_plot.reset_index()

cp_sample = to_plot.copy()
cp_sample = cp_sample.drop(columns=["port"])
cp_sample = cp_sample.set_index("type")


def ts_latex_table_delevered(full_sample, cp_sample, save_name: str, num_format="%.4f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    table = Tabular("".join(["l"] + ["c"] * (full_sample.shape[1] - 1) + ["c"]), booktabs=True)
    table.add_row(
        [
            "",
            MultiColumn(2, align="c", data=math(full_sample.columns.get_level_values(0)[0])),
            "",
            MultiColumn(2, align="c", data=math(full_sample.columns.get_level_values(0)[3])),
            "",
            MultiColumn(2, align="c", data=math(full_sample.columns.get_level_values(0)[-1])),
        ]
    )
    to_add = [""] + ["" if "EMPTY" in c else c for c in full_sample.columns.get_level_values(1).tolist()]
    table.add_row([MultiColumn(1, align="c", data=c) for c in to_add])
    table.add_hline(start=2, end=3)
    table.add_hline(start=5, end=6)
    table.add_hline(start=8, end=9)

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


ts_latex_table_delevered(full_sample, cp_sample, f"trading_strat_vw_{add_on}", num_format="%.3f")


# %%
# LARGEST 500 RELATIVE TO CRSP UNIVERSE
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
add_on = "largest"
prediction_dict = load(os.path.join(modelLoc, f"prediction_dict_{add_on}.pkl"))


# %% OOS R2 full sample
to_plot = []
for i, model in enumerate(["N-En"]):
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


# ---- TS R2
calls = TS_R2(to_plot[to_plot["type"] == "call"].drop(columns=["type"]))
calls = calls.to_frame()
calls["type"] = "Call"
puts = TS_R2(to_plot[to_plot["type"] == "put"].drop(columns=["type"]))
puts = puts.to_frame()
puts["type"] = "Put"
ts_r2 = TS_R2(to_plot.drop(columns=["type"]))
ts_r2 = ts_r2.to_frame()
ts_r2["type"] = "All"
ts_r2 = pd.concat((calls, puts, ts_r2), axis=0)
ts_r2 = ts_r2.reset_index()
ts_r2.columns = ["model", "score", "type"]

cw_test = ClarkWest(to_plot.drop(columns=["type"]), 12, benchmark_type="zero", cw_adjust=False)
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


# after 2009
calls = TS_R2(to_plot.loc[(to_plot.index >= "2010-01") & (to_plot["type"] == "call")].drop(columns=["type"]))
calls = calls.to_frame()
calls["type"] = "Call"
puts = TS_R2(to_plot.loc[(to_plot.index >= "2010-01") & (to_plot["type"] == "put")].drop(columns=["type"]))
puts = puts.to_frame()
puts["type"] = "Put"
tmp = TS_R2(to_plot.loc["2010":].drop(columns=["type"]))
tmp = tmp.to_frame()
tmp["type"] = "All"
tmp = pd.concat((calls, puts, tmp), axis=0)
tmp = tmp.reset_index()
tmp.columns = ["model", "score", "type"]
tmp["model"] = "N-En2010"

ts_r2 = pd.concat((ts_r2, tmp))

cw_test = ClarkWest(to_plot.loc["2010":].drop(columns=["type"]), 12, benchmark_type="zero", cw_adjust=False)
tmp = []
for c in cw_test:
    if c > 3.09:
        tmp.append("***")
    elif c > 2.326:
        tmp.append("**")
    elif c > 1.645:
        tmp.append("*")
    else:
        tmp.append("")
tmp = pd.Series(data=tmp, index=cw_test.index)


stars = pd.concat((stars, tmp))


# ---- XS R2
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


# after 2009
calls = XS_R2(to_plot.loc[(to_plot.index >= "2010-01") & (to_plot["type"] == "call")].drop(columns=["type"]))
calls = calls.to_frame()
calls["type"] = "Call"
puts = XS_R2(to_plot.loc[(to_plot.index >= "2010-01") & (to_plot["type"] == "put")].drop(columns=["type"]))
puts = puts.to_frame()
puts["type"] = "Put"
tmp = XS_R2(to_plot.drop(columns=["type"]))
tmp = tmp.to_frame()
tmp["type"] = "All"
tmp = pd.concat((calls, puts, tmp), axis=0)
tmp = tmp.reset_index()
tmp.columns = ["model", "score", "type"]
tmp["model"] = "N-En2010"

xs_r2 = pd.concat((xs_r2, tmp))


cw_test = ClarkWest(to_plot.loc["2010":].drop(columns=["type"]), 12, benchmark_type="xs", cw_adjust=False)
tmp = []
for c in cw_test:
    if c > 3.09:
        tmp.append("***")
    elif c > 2.326:
        tmp.append("**")
    elif c > 1.645:
        tmp.append("*")
    else:
        tmp.append("")
tmp = pd.Series(data=tmp, index=cw_test.index)

xs_stars = pd.concat((xs_stars, tmp))


ts_r2["model"] = ts_r2["model"].replace({"N-En": "Full Sample", "N-En2010": "2010 and after"})
xs_r2["model"] = xs_r2["model"].replace({"N-En": "Full Sample", "N-En2010": "2010 and after"})
fig, axes = plt.subplots(1, 2, figsize=(width, height * 0.8), dpi=1000, sharey=True)
sns.barplot(x="model", y="score", hue="type", data=ts_r2, ax=axes[0], palette=sns_palette(3))
sns.barplot(x="model", y="score", hue="type", data=xs_r2, ax=axes[1], palette=sns_palette(3))
for i_ax, ax in enumerate(axes):
    ax.set_ylabel("")
    ax.set_xlabel("")
    # ax.axhline(0, ls="-", lw=1, color="k")
    ax.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax.set_axisbelow(True)
    if i_ax == 0:
        for i, (_, txt) in enumerate(stars.iteritems()):
            ax.annotate(txt, (i, 0.0), ha="center", color="white")
    else:
        for i, (_, txt) in enumerate(xs_stars.iteritems()):
            ax.annotate(txt, (i, 0.0), ha="center", color="white")
axes[1].legend([], frameon=False, title=None)
axes[0].legend(frameon=False, title=None)
fig.tight_layout()
fig.savefig(f"../08_figures/r2_{add_on}.pdf", bbox_inches="tight")


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
fig.savefig(f"../08_figures/ts_over_time_{add_on}.pdf")


# %%
# Trading strategy using **value-weights*:
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

    # full sample 2003 -- 2020
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

    # after 2009
    output = []
    for col in ["predicted", "target"]:
        returns = tmp.loc["2010":, col].unstack()
        returns.columns = "Lo 2 3 4 5 6 7 8 9 Hi".split()
        returns["H-L"] = returns["Hi"] - returns["Lo"]
        returns = returns.stack()
        returns.name = col
        returns.index.names = ["date", "port"]
        output.append(returns)
    output = pd.concat(output, axis=1)
    model_returns[model + "2010"] = output["target"].copy()

    grouper = output.groupby("port")
    strat = grouper.mean()
    strat.columns = ["Pred", "Avg"]
    strat["SD"] = grouper.target.std().tolist()
    strat *= 100
    strat["SR"] = strat["Avg"] / strat["SD"]
    strat.columns = pd.MultiIndex.from_arrays([[model + " 2010 and after"] * strat.shape[1], strat.columns])
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
to_plot["EMPTY_2"] = ""

full_sample = to_plot.copy()


# ---- split by puts and calls (only hml)
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

    # full sample
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

    # after 2009
    output = []
    for target_col in ["predicted", "target"]:
        # returns = tmp.groupby(["date", col, "port"])[target_col].mean()
        returns = tmp.loc["2010":, target_col].copy()
        returns = returns.unstack()
        returns.columns = "Lo 2 3 4 5 6 7 8 9 Hi".split()
        returns["H-L"] = returns["Hi"] - returns["Lo"]
        returns = returns.stack()
        returns.name = target_col
        returns.index.names = ["date", col, "port"]
        output.append(returns)
    output = pd.concat(output, axis=1)
    output = output[output.index.get_level_values("port") == "H-L"]
    model_returns[model + "2010"] = output["target"].copy()

    grouper = output.groupby([col, "port"])
    strat = grouper.mean()
    strat.columns = ["Pred", "Avg"]
    strat["SD"] = grouper.target.std().tolist()
    strat *= 100
    strat["SR"] = strat["Avg"] / strat["SD"]
    strat.columns = pd.MultiIndex.from_arrays([[model + "2010 and after"] * strat.shape[1], strat.columns])
    to_plot.append(strat)

model_returns = pd.concat(model_returns, axis=1)

to_plot = pd.concat(to_plot, axis=1)

to_plot.insert(4, "EMPTY_0", "")
to_plot["EMPTY_1"] = ""
to_plot["EMPTY_2"] = ""
to_plot = to_plot.reset_index()

cp_sample = to_plot.copy()
cp_sample = cp_sample.drop(columns=["port"])
cp_sample = cp_sample.set_index("type")


ts_latex_table(full_sample, cp_sample, f"trading_strat_vw_{add_on}", num_format="%.3f")


# %%
# LARGEST 500 RELATIVE TO CRSP UNIVERSE BUT FULL MODEL
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
add_on = "largest_full_model"
prediction_dict = load(os.path.join(modelLoc, "prediction_dict_comparison.pkl"))


# %% OOS R2 full sample
to_plot = []
for i, model in enumerate(["L-En", "N-En"]):
    if i == 0:
        target = prediction_dict[model]["predictions"][["optionid", "permno", "target"]].copy()
    tmp = prediction_dict[model]["predictions"]["predicted"].copy()
    tmp.name = model
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot = pd.concat((to_plot, target), axis=1)


# restrict to largest 500 CRSP stocks
crsp = pd.read_parquet(
    "../03_data/char_stocks/crsp_prices.pq",
    columns=["permno", "date", "close", "shrout"],
)
crsp["mcap"] = crsp["shrout"] * np.abs(crsp["close"])
crsp["mcap_rank"] = crsp.groupby("date")["mcap"].rank(ascending=False)
crsp["date"] = crsp["date"].dt.to_period("M")

# drop columns:
crsp = crsp[["permno", "date", "mcap_rank"]]

# merge
to_plot = to_plot.reset_index()
to_plot = to_plot.merge(crsp, on=["date", "permno"])
to_plot = to_plot[to_plot["mcap_rank"] <= 500]
to_plot = to_plot.drop(columns=["mcap_rank"])
to_plot = to_plot.set_index(["date"])
to_plot = to_plot.drop(columns=["permno"])


# add type (call/put) information:
to_plot = to_plot.reset_index()
to_plot = to_plot.merge(class_groups[["date", "optionid", "type"]], on=["date", "optionid"])
to_plot = to_plot.drop(columns=["optionid"])
to_plot = to_plot.set_index("date")


# ---- TS R2
calls = TS_R2(to_plot[to_plot["type"] == "call"].drop(columns=["type"]))
calls = calls.to_frame()
calls["type"] = "Call"
puts = TS_R2(to_plot[to_plot["type"] == "put"].drop(columns=["type"]))
puts = puts.to_frame()
puts["type"] = "Put"
ts_r2 = TS_R2(to_plot.drop(columns=["type"]))
ts_r2 = ts_r2.to_frame()
ts_r2["type"] = "All"
ts_r2 = pd.concat((calls, puts, ts_r2), axis=0)
ts_r2 = ts_r2.reset_index()
ts_r2.columns = ["model", "score", "type"]

cw_test = ClarkWest(to_plot.drop(columns=["type"]), 12, benchmark_type="zero", cw_adjust=False)
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


# ---- XS R2
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


fig, axes = plt.subplots(1, 2, figsize=(width, height * 0.8), dpi=1000, sharey=True)
sns.barplot(x="model", y="score", hue="type", data=ts_r2, ax=axes[0], palette=sns_palette(3))
sns.barplot(x="model", y="score", hue="type", data=xs_r2, ax=axes[1], palette=sns_palette(3))
for i_ax, ax in enumerate(axes):
    ax.set_ylabel("")
    ax.set_xlabel("")
    # ax.axhline(0, ls="-", lw=1, color="k")
    ax.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax.set_axisbelow(True)
    if i_ax == 0:
        for i, (_, txt) in enumerate(stars.iteritems()):
            ax.annotate(txt, (i, 0.0), ha="center", color="white")
    else:
        for i, (_, txt) in enumerate(xs_stars.iteritems()):
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

    # restrict to largest 500 CRSP stocks
    crsp = pd.read_parquet(
        "../03_data/char_stocks/crsp_prices.pq",
        columns=["permno", "date", "close", "shrout"],
    )
    crsp["mcap"] = crsp["shrout"] * np.abs(crsp["close"])
    crsp["mcap_rank"] = crsp.groupby("date")["mcap"].rank(ascending=False)
    crsp["date"] = crsp["date"].dt.to_period("M")

    # drop columns:
    crsp = crsp[["permno", "date", "mcap_rank"]]

    # merge
    tmp = tmp.reset_index()
    tmp = tmp.merge(crsp, on=["date", "permno"])
    tmp = tmp[tmp["mcap_rank"] <= 500]
    tmp = tmp.drop(columns=["mcap_rank"])
    tmp = tmp.set_index(["date"])
    tmp = tmp.drop(columns=["permno"])

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

    # restrict to largest 500 CRSP stocks
    crsp = pd.read_parquet(
        "../03_data/char_stocks/crsp_prices.pq",
        columns=["permno", "date", "close", "shrout"],
    )
    crsp["mcap"] = crsp["shrout"] * np.abs(crsp["close"])
    crsp["mcap_rank"] = crsp.groupby("date")["mcap"].rank(ascending=False)
    crsp["date"] = crsp["date"].dt.to_period("M")

    # drop columns:
    crsp = crsp[["permno", "date", "mcap_rank"]]

    # merge
    tmp = tmp.reset_index()
    tmp = tmp.merge(crsp, on=["date", "permno"])
    tmp = tmp[tmp["mcap_rank"] <= 500]
    tmp = tmp.drop(columns=["mcap_rank"])
    tmp = tmp.set_index(["date"])
    tmp = tmp.drop(columns=["permno"])

    tmp = tmp.merge(class_groups[["date", "optionid", col]], on=["date", "optionid"])

    tmp["port"] = tmp.groupby(["date", col]).predicted.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    tmp = tmp.set_index("date")

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

ts_latex_table(full_sample, cp_sample, f"trading_strat_vw_{add_on}", num_format="%.3f")


# %%
