# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %% Packages:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from numba import njit
import glob
from pandas.tseries.offsets import BDay
from PrepareCovariates.helper.map_r_rate import map_r_rate
from scipy.stats import norm

from pylatex import Tabular, MultiColumn, Math
from pylatex.utils import NoEscape

from analysis_setup import prediction_dict, sns_palette, width, height, analysisLoc
from scoring import ClarkWest, XS_R2, TS_R2


class_groups = pd.read_parquet(os.path.join(analysisLoc, "class_groups.pq"))
class_groups_info = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups_and_info.pq"),
    columns=[
        "date",
        "permno",
        "optionid",
        "ttm",
        "moneyness",
        "bucket",
        "C",
        "close",
        "mid",
        "baspread",
        "idiovol",
        "beta",
        "ill",
        "inst_share",
        "num_analysts",
        "leverage",
        "optspread",
        "margin_denominator",
        "ivrv",
        "doi",
    ],
)
option_flows = pd.read_parquet(os.path.join(analysisLoc, "option_flows_with_info.pq"))
option_flows_permno = option_flows.groupby(["date", "permno"]).sum()
option_flows_permno = option_flows_permno.reset_index()
option_flows_permno = option_flows_permno.drop(columns=["optionid"])

DUAL_PLOTS = False


# %% Setup:
samples = {
    "mispricing_score": "SY Mispricing Score",
    "opt_mispricing": "Option Mispricing Score",
    "combined_mispricing": "Combined Mispricing Score",
    "arb": "Stock Arbitrage Score",
    "opt_arb": "Option Arbitrage Score",
    "combined_arb": "Combined Arbitrage Score",
    "mispricing_q5": "Option Mispricing",
    "inst_share_quintile": "Institutional Share",
    "analyst_quintile": "Analyst Coverage",
    "split_inst_share_quintile": "Institutional Holdings",
    "split_analyst_quintile": "Analyst Coverage",
    "split_inst_share": "Institutional Holdings",
    "split_analyst_coverage": "Analyst Coverage",
    "split_stock_liquidity": "Stock Illiquidity",
    "split_option_liquidity": "Option Illiquidity",
    "split_option_liquidity_permno": "Option Illiquidity",
    "high_inst_share": "High\nInst",
    "low_inst_share": "Low\nInst",
    "analyst_coverage": "High\nAnalyst",
    "no_analyst_coverage": "Low\nAnalyst",
}

column_names = {
    "Institutional Holdings": "High Inst. Share",
    "Analyst Coverage": "High Analyst Cvg.",
    "Option Illiquidity": "Illiquid Options",
    "Stock Illiquidity": "Illiquid Underlying",
}


# %% Prepare data routines:
def quantile_cuts(data, col, n_port, port_names: list):
    qs = data.groupby("date")[col].apply(lambda x: pd.qcut(x, n_port, labels=False, duplicates="drop"))
    names = {}
    for i, name in enumerate(port_names):
        names[i] = name
    qs = qs.fillna("no_data")
    qs = qs.replace(names)
    return qs


def bsm(S, K, r, vol, ttm, call_flag):
    @njit
    def d1(S, K, r, vol, ttm):
        return (np.log(S / K) + (r + vol**2 / 2) * ttm) / (vol * np.sqrt(ttm))

    @njit
    def d2(d1_val, vol, ttm):
        return d1_val - vol * np.sqrt(ttm)

    def phi(x):
        return norm.cdf(x)

    d1_val = d1(S, K, r, vol, ttm)
    d2_val = d2(d1_val, vol, ttm)
    return (S * phi(d1_val * call_flag) - K * np.exp(-r * ttm) * phi(d2_val * call_flag)) * call_flag


@njit
def vw_mean(target, weight):
    return (target * weight).sum() / weight.sum()


def return_option_flows(data_for_flows: pd.DataFrame, volume_type: str, investor_groups: list):
    flows = data_for_flows[
        [
            c
            for c in data_for_flows.columns
            if c.endswith("_" + volume_type) and any([c.startswith(group) for group in investor_groups])
        ]
    ].sum(axis=1)
    return flows


def prepare_model_data(model, cols_to_use, start_date=None, skip_renaming: bool = False):
    model_table = {}
    class_errors = prediction_dict[model]["errors"].copy()
    for c_error in cols_to_use:
        print(c_error)
        tmp = class_errors[c_error]
        if start_date:
            tmp = tmp[tmp.index.get_level_values("date") >= start_date]
        scores = (
            tmp.groupby(c_error)[
                [c for c in tmp.columns if c.startswith("m_") and "-" not in c] + ["predicted", "target"]
            ]
            .apply(lambda x: 1 - x.sum() / x["target"].sum())
            .drop(columns=["target"])
        )
        predicted = scores["predicted"]
        scores = scores.subtract(scores["predicted"], axis=0).div(scores["predicted"].abs(), axis=0)
        scores = scores.drop(columns=["predicted"])
        scores = scores.div(scores.sum(axis=1), axis=0)
        scores.columns = [c.strip("m_") for c in scores.columns]
        scores = pd.concat((predicted, scores), axis=1)
        # scores.columns = [group_dict[c] for c in scores.columns]
        scores = scores[scores.index != "no_data"]
        if not skip_renaming:
            scores["Sample"] = [samples[idx.replace("_permno", "")] for idx in scores.index]
        else:
            scores["Sample"] = scores.index
        scores = scores.set_index("Sample")
        if "inv" in c_error:
            scores.index = [i.replace("V", "I") for i in scores.index]
        model_table[samples[c_error.replace("_permno", "")]] = scores
    return pd.concat(model_table)


def obtain_stars(cw_test):
    stars = []
    for c in cw_test.values:
        if c > 3.09:
            stars.append("***")
        elif c > 2.326:
            stars.append("**")
        elif c > 1.645:
            stars.append("*")
        else:
            stars.append("")
    stars = pd.Series(data=stars, index=cw_test.index)
    return stars


def ts_payoff_high_low(predictions, n_port=3):
    predictions["port"] = predictions.groupby("date")["predicted"].apply(
        lambda x: pd.qcut(x, n_port, labels=False, duplicates="drop")
    )
    tmp = predictions.groupby(["date", "port"])["target"].mean().unstack()
    tmp = tmp[n_port - 1] - tmp[0]
    out = pd.DataFrame(index=[0])
    out["Avg"] = tmp.mean()
    out["SR"] = tmp.mean() / tmp.std()
    return out


def ts_payoff(predictions, n_port=3):
    predictions["port"] = predictions.groupby("date")["predicted"].apply(
        lambda x: pd.qcut(x, n_port, labels=False, duplicates="drop")
    )
    tmp = predictions.groupby(["date", "port"])["target"].mean().unstack()
    return tmp


def feature_imp_pairs_plot(data, save_name):
    # data = feature_tables["N-En"] * 100
    # data = data.T
    data.index.name = "Group"
    minimum = data.min().min()
    maximum = data.max().max()
    data = data.loc[data.mean(axis=1).sort_values(ascending=False).index]
    # data = data.loc[data.iloc[:, 0].sort_values(ascending=False).index]
    # data = data.sort_values("Group")
    data = data.reset_index()

    # sns.set(font_scale=0.8)
    g = sns.PairGrid(
        data,
        x_vars=data.drop(columns=["Group"]).columns,
        y_vars=["Group"],
        height=height * 0.8,
        aspect=(height * 0.8) / (data.shape[1] - 1) / 1.6,
    )

    # Draw a dot plot using the stripplot function
    g.map(
        sns.stripplot,
        size=10,
        orient="h",
        jitter=False,
        palette="hls",
        linewidth=1,
        edgecolor="w",
    )

    # Use the same x axis limits on all columns and add better labels
    g.set(xlim=(np.round(minimum, 0) - 3, np.round(maximum, 0) + 3), xlabel="", ylabel="")

    # Use semantically meaningful titles for the columns
    titles = data.drop(columns=["Group"]).columns.tolist()

    for ax, title in zip(g.axes.flat, titles):
        # Set a different title for each axes
        ax.set(title=title)

        # Make the grid horizontal instead of vertical
        # ax.grid(which="both", ls="--")
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    sns.despine(left=True, bottom=True)
    for ax in g.axes[0]:
        ax.axvline(0, ls="--", lw=0.5)
    g.fig.tight_layout()
    g.fig.savefig("../08_figures/%s.pdf" % save_name)
    return g


def friction_tables_and_graphs(cols_to_use: list, name: str):
    r2s = []
    for col in cols_to_use:
        r2 = predictions.groupby(col).apply(lambda x: TS_R2(x[["target", "predicted"]]))
        r2 = r2.loc[~r2.index.isin(["no_data", "nan"])]
        r2["Sample"] = samples[r2.index.name]
        r2.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        r2.index.name = "Portfolio"
        r2 = r2.reset_index()
        r2s.append(r2)
    r2s = pd.concat(r2s).squeeze()

    stars = []
    for col in cols_to_use:
        print(col)
        tmp = predictions.groupby(col)[["target", "predicted"]].apply(
            lambda x: ClarkWest(x, 12, benchmark_type="zero", cw_adjust=False)
        )
        tmp = tmp.loc[~tmp.index.isin(["no_data", "nan"])]
        tmp["Sample"] = samples[tmp.index.name]
        tmp.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        tmp.index.name = "Portfolio"
        tmp = tmp.reset_index()
        tmp = tmp.set_index(["Sample", "Portfolio"])
        stars.append(obtain_stars(tmp))
    stars = pd.concat(stars)

    fig, ax = plt.subplots(figsize=(width, height * 0.6), sharex=True)
    plots = sns.barplot(x="Sample", y="predicted", hue="Portfolio", data=r2s, palette=sns_palette(5))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax.set_axisbelow(True)
    for i, bar in enumerate(plots.patches):
        print(bar)
        plots.annotate(
            stars.iloc[i // 2 + 5 * (i % 2)] if r2s.Sample.nunique() > 1 else stars.iloc[i],
            (bar.get_x() + bar.get_width() / 2, r2s["predicted"].max()),
            ha="center",
            va="center",
            size=8,
            xytext=(0, 2),
            textcoords="offset points",
        )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"../08_figures/{name}_r2.pdf")

    # XS R2:
    r2s = []
    for col in cols_to_use:
        r2 = predictions.groupby(col).apply(lambda x: XS_R2(x[["target", "predicted"]]))
        r2 = r2.loc[~r2.index.isin(["no_data", "nan"])]
        r2["Sample"] = samples[r2.index.name]
        r2.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        r2.index.name = "Portfolio"
        r2 = r2.reset_index()
        r2s.append(r2)
    r2s = pd.concat(r2s).squeeze()

    stars = []
    for col in cols_to_use:
        print(col)
        tmp = predictions.groupby(col)[["target", "predicted"]].apply(
            lambda x: ClarkWest(x, 12, benchmark_type="zero", cw_adjust=False)
        )
        tmp = tmp.loc[~tmp.index.isin(["no_data", "nan"])]
        tmp["Sample"] = samples[tmp.index.name]
        tmp.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        tmp.index.name = "Portfolio"
        tmp = tmp.reset_index()
        tmp = tmp.set_index(["Sample", "Portfolio"])
        stars.append(obtain_stars(tmp))
    stars = pd.concat(stars)

    fig, ax = plt.subplots(figsize=(width, height * 0.6), sharex=True)
    plots = sns.barplot(x="Sample", y="predicted", hue="Portfolio", data=r2s, palette=sns_palette(5))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
    ax.set_axisbelow(True)
    for i, bar in enumerate(plots.patches):
        print(bar)
        plots.annotate(
            stars.iloc[i // 2 + 5 * (i % 2)] if r2s.Sample.nunique() > 1 else stars.iloc[i],
            (bar.get_x() + bar.get_width() / 2, r2s["predicted"].max()),
            ha="center",
            va="center",
            size=8,
            xytext=(0, 2),
            textcoords="offset points",
        )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"../08_figures/{name}_xs_r2.pdf")

    # Trading strat table:
    returns = {}
    tstats = {}
    stars = {}
    n_port = 5
    for col in cols_to_use:
        tmp = predictions[predictions[col] != "nan"].groupby(col).apply(ts_payoff, n_port)
        tmp = tmp.reset_index()
        tmp[col] = tmp[col].astype("f4")
        tmp = tmp.set_index([col, "date"])
        tmp["H-L"] = tmp[n_port - 1] - tmp[0]
        hml = tmp.loc[n_port - 1] - tmp.loc[0]
        hml[col] = "H-L"
        hml = hml.reset_index().set_index([col, "date"])
        tmp = pd.concat((tmp, hml))

        means = tmp.groupby(col).mean() * 100
        means.columns = ["Low Pred.", "2", "3", "4", "High Pred.", "H-L"]
        means.index = ["Low", "2", "3", "4", "High", "H-L"]
        ts = (
            tmp.groupby(col)
            .apply(
                lambda x: x.apply(
                    lambda y: sm.OLS(y.dropna(), np.ones(y.dropna().shape)).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues
                )
            )
            .reset_index(level=1, drop=True)
        )
        ts.columns = [r"$\text{Low} \hat{r}$", "2", "3", "4", r"$\text{High} \hat{r}$", "H-L"]
        ts.index = ["Low", "2", "3", "4", "High", "H-L"]
        star = pd.DataFrame(index=ts.index, columns=ts.index, data="")
        star[(ts.abs() > 1.65).values] = "*"
        star[(ts.abs() > 1.96).values] = "**"
        star[(ts.abs() > 2.56).values] = "***"

        returns[samples[col]] = means
        tstats[samples[col]] = ts
        stars[samples[col]] = star

    def latex_table(returns, stars, save_name: str, num_format="%.4f"):
        def math(x):
            return Math(data=[NoEscape(x)], inline=True)

        first_key = list(returns.keys())[0]
        N = 1 + returns[first_key].shape[1]
        table = Tabular(
            "".join(["l"] + ["r"] * (N - 1)),
            booktabs=True,
        )
        print(table)

        to_add = [""]
        for d in returns[first_key].columns.tolist():
            if "$" in d:
                to_add.append(MultiColumn(1, align="c", data=math(d)))
            else:
                to_add.append(MultiColumn(1, align="c", data=d))
        table.add_row(to_add)
        table.add_hline()

        for i_key, key in enumerate(returns.keys()):
            table.add_row([MultiColumn(N, align="c", data=key)])
            table.add_hline()

            for i, (idx, vals) in enumerate(returns[key].iterrows()):
                to_add = [idx]
                for j, v in enumerate(vals):
                    to_add.append(num_format % v + stars[key].iloc[i, j])
                table.add_row(to_add)

            if i_key != (len(returns.keys()) - 1):
                table.add_hline()
                table.add_empty_row()

        table.generate_tex("../08_figures/%s" % save_name)

    file = name
    latex_table(returns, stars, file, "%.3f")

    with open("../08_figures/" + file + ".tex") as f:
        lines = f.readlines()

    lines[0] = lines[0].replace(
        "l" + "r" * (n_port + 1),
        r"l!{\quad}" + r"S[table-format = 2.3,table-space-text-post={*}]!{\quad}" * (n_port + 1),
    )

    with open("../08_figures/" + file + ".tex", "w") as f:
        f.writelines(lines)

# %%
# ----------------- COMPOSITE ARBITRAGE INDEX -----------------------------
flag_atm_only = False
flag_mean = False
index_q = 10
pf_q = 5

# ----------------- COMPOSITE ARBITRAGE INDEX STOCK-LEVEL -----------------------------

# ---- obtain raw characteristics
char_files = glob.glob("../04_results/option_sample/characteristics*.pq")
char = pd.concat(
    [pd.read_parquet(file, columns=["date", "optionid", "age", "mve", "bucket"]) for file in char_files]
).reset_index()
char["date"] = char["date"].dt.to_period("M")

if flag_atm_only:
    char = char.loc[char.bucket == "short_term_atm", :]

# ---- Turan's index + some other stuff
cols_to_use = ["idiovol", "inst_share", "num_analysts"]
arb_stock = class_groups_info[["date", "optionid", "permno"] + cols_to_use]
arb_stock = arb_stock.merge(char[["date", "optionid", "age", "mve"]], on=["date", "optionid"])
arb_stock[["inst_share", "num_analysts", "age", "mve"]] *= -1  # higher values --> lower arb cost
arb_stock = arb_stock.dropna()
arb_stock = arb_stock.groupby(["date", "permno"]).first()
arb_stock = arb_stock.groupby("date").rank(method="dense", pct=True) * 100

if flag_mean:
    arb_stock = arb_stock.mean(axis=1)
else:
    arb_stock = arb_stock.groupby(["date"]).transform(lambda x: pd.qcut(x, q=index_q, labels=False, duplicates="drop"))
    arb_stock = arb_stock.sum(axis=1)

arb_stock.name = "arb"

# ----------------- COMPOSITE ARBITRAGE INDEX OPTION-LEVEL -----------------------------

# ---- obtain raw characteristics
char_files = glob.glob("../04_results/option_sample/characteristics*.pq")
char = pd.concat(
    [
        pd.read_parquet(
            file,
            columns=[
                "date",
                "optionid",
                "optspread",  # illiquidity
                "margin_denominator",
                "bucket_dvol",  # volume
                "close",
                "ivvol",
                "oi",
                "hkurt",
                "baspread",
            ],
        )
        for file in char_files
    ]
).reset_index()

char["date"] = char["date"].dt.to_period("M")

permnos = class_groups_info[["date", "permno", "optionid", "bucket"]]
char = char.merge(permnos, on=["date", "optionid"])

if flag_atm_only:
    char = char.loc[char.bucket == "short_term_atm", :]

arb = char.drop(columns=["optionid"]).groupby(["date", "permno"])[["hkurt", "baspread", "ivvol"]].first()

arb = arb.groupby("date").rank(method="dense", pct=True) * 100
if ~flag_mean:
    arb = arb.groupby(["date"]).transform(lambda x: pd.qcut(x, q=index_q, labels=False, duplicates="drop"))

# add option-level arb stuff
opt_arb = char[["date", "optionid", "permno", "margin_denominator", "oi", "close"]]

opt_arb["margin_denominator"] = opt_arb["margin_denominator"] / opt_arb["close"]  # make comparable in XS
opt_arb = opt_arb.drop(columns=["close"])

opt_arb[["oi"]] *= -1  # higher values --> lower arb cost

opt_arb = opt_arb.set_index(["date", "optionid", "permno"])
opt_arb = opt_arb.groupby("date").rank(method="dense", pct=True) * 100
if ~flag_mean:
    opt_arb = opt_arb.groupby(["date"]).transform(lambda x: pd.qcut(x, q=index_q, labels=False, duplicates="drop"))


# add bucket-level arb stuff
bucket_arb = char[["date", "optionid", "permno", "bucket_dvol"]]
bucket_arb = bucket_arb.sort_values(["permno", "date"])
bucket_arb[["bucket_dvol"]] *= -1  # higher values --> lower arb costs
bucket_arb = bucket_arb.set_index(["date", "optionid", "permno"])
bucket_arb = bucket_arb.groupby("date").rank(method="dense", pct=True) * 100
if ~flag_mean:
    bucket_arb = bucket_arb.groupby(["date"]).transform(lambda x: pd.qcut(x, q=index_q, labels=False, duplicates="drop"))


# merge stock and option-level stuff
arb = arb.merge(
    opt_arb.reset_index("optionid"),
    on=["date", "permno"],
)
arb = arb.set_index("optionid", append=True)
arb = arb.merge(bucket_arb, on=["date", "permno", "optionid"])
print(f"Using values for {arb.columns} in the composite mispricing index.")
if flag_mean:
    arb = arb.mean(axis=1)
else:
    arb = arb.sum(axis=1)


arb.name = "opt_arb"
arb = arb.to_frame()
arb = arb.reset_index()
arb = arb.merge(arb_stock, on=["date", "permno"], how="left")
arb["combined_arb"] = arb[["opt_arb", "arb"]].mean(axis=1, skipna=False)
arb = arb.set_index(["date", "optionid"])


arb["arb"] = arb.groupby("date").arb.apply(lambda x: pd.qcut(x, q=pf_q, labels=False, duplicates="drop"))
arb["opt_arb"] = arb.groupby("date").opt_arb.apply(lambda x: pd.qcut(x, q=pf_q, labels=False, duplicates="drop"))
arb["combined_arb"] = arb.groupby("date").combined_arb.apply(lambda x: pd.qcut(x, q=pf_q, labels=False, duplicates="drop"))


predictions = prediction_dict["N-En"]["predictions"]
predictions = predictions.merge(arb, on=["date", "optionid"], how="left")

file_names = "arb"
if flag_atm_only:
    file_names = file_names + "_shortterm_atm"

if flag_mean:
    file_names = file_names + "_mean"

friction_tables_and_graphs(["arb", "opt_arb"], file_names)
friction_tables_and_graphs(["combined_arb"], "combined" + file_names)


# %%
# ----------------- COMPOSITE MISPRICING INDEX -----------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# ----------------- STOCK-LEVEL -----------------------------

# ---- Stambaugh Yuan
mispricing = pd.read_csv("../03_data/Misp_Score.txt", delimiter="  ")
mispricing.columns = ["permno", "yyyymm", "mispricing_score"]
mispricing["yyyymm"] = pd.to_datetime(mispricing["yyyymm"], format="%Y%m").dt.to_period("M")
mispricing = mispricing.rename(columns={"yyyymm": "date"})
mispricing = mispricing.set_index(["date", "permno"])
mispricing["mispricing_score"] = mispricing.groupby("date").mispricing_score.apply(
    lambda x: pd.qcut(x, q=index_q, labels=False, duplicates="drop")
)


# ---- obtain mispricing of options
# get rv per permno/date for 1Q lookback window:
taq_files = glob.glob("../03_data/taq/*.pq")
taq_data = []
for file in taq_files:
    data = pd.read_parquet(file, columns=["datetime", "permno", "r2_to_close", "r2_from_prev_close"])
    data["r2"] = data["r2_to_close"] + data["r2_from_prev_close"]
    data = data.drop(columns=["r2_to_close", "r2_from_prev_close"])
    data = data.set_index("datetime")
    data.r2 *= 252  # annual rv
    taq_data.append(data)
taq_data = pd.concat(taq_data)

taq_data.index = taq_data.index.to_period("M")
rolling_rv = taq_data.groupby(["permno", "datetime"]).mean()
rolling_rv = rolling_rv.groupby("permno").apply(lambda x: x.rolling(3, min_periods=3).mean())
rolling_rv = rolling_rv.dropna()
rolling_rv.index.names = ["permno", "date"]
rolling_rv = rolling_rv.reset_index()
rolling_rv["date"] = rolling_rv["date"].dt.to_timestamp()

atm_options = class_groups_info[class_groups_info.bucket == "short_term_atm"]
atm_options["date"] = atm_options.date.dt.to_timestamp()
atm_options["exdate"] = atm_options["date"] + pd.to_timedelta(atm_options["ttm"], "D")
atm_options["ttm"] /= 365
atm_options["C"] = atm_options["C"].replace({1: 1, 0: -1})
atm_options["K"] = atm_options["moneyness"] * atm_options["close"]

# get r
r = pd.read_csv("../03_data/yield_curve.zip", index_col=0, parse_dates=True)
r.rate /= 100
r.columns = ["days", "r"]
r.days /= 365
bdays = pd.bdate_range(r.index.min(), r.index.max())
bdays_to_append = bdays[~bdays.isin(r.index)]
for b in bdays_to_append:
    to_append = r[r.index == b - BDay()]
    to_append.index += BDay()
    r = r.append(to_append)
r = r.reset_index().sort_values(["date", "days"]).set_index("date")
r = r.groupby([r.index.to_period("M"), "days"]).mean()
r = r.reset_index(level="days")
r.index = r.index.to_timestamp()
r_mapped = atm_options.groupby("date", sort=False).apply(lambda x: map_r_rate(x, r, "exdate", "ttm"))
atm_options["r"] = r_mapped.values

# merge rv
atm_options = atm_options.merge(rolling_rv, on=["date", "permno"])

# get "theoretical" price
atm_options["theoretical_price"] = bsm(
    atm_options["close"].to_numpy(),
    atm_options["K"].to_numpy(),
    atm_options["r"].to_numpy(),
    np.sqrt(atm_options["r2"]).to_numpy(),
    atm_options["ttm"].to_numpy(),
    atm_options["C"].to_numpy(),
)

# get mispricing
atm_options["opt_mispricing"] = np.log(atm_options["mid"] / atm_options["theoretical_price"])
opt_mispricing = atm_options.groupby(["date", "permno"])["opt_mispricing"].mean().abs()  # absolute opt_mispricing
opt_mispricing = opt_mispricing.reset_index()
opt_mispricing["date"] = opt_mispricing["date"].dt.to_period("M")


arb = class_groups_info[["date", "permno", "ivrv"]]
arb = arb.merge(opt_mispricing, on=["date", "permno"], how="outer")
arb = arb.groupby(["date", "permno"]).first()
arb = arb.groupby("date").rank(method="dense", pct=True) * 100


# ---- add absolute return prediction of our model as measure of "mispricing"
opt_arb = prediction_dict["N-En"]["predictions"][["optionid", "permno", "predicted"]].copy().reset_index()
opt_arb["predicted"] = opt_arb["predicted"].abs()
opt_arb = opt_arb.set_index(["date", "optionid", "permno"])
opt_arb = opt_arb.groupby("date").rank(method="dense", pct=True) * 100


arb = arb.merge(
    opt_arb.reset_index("optionid"),
    on=["date", "permno"],
)
arb = arb.set_index("optionid", append=True)

arb = arb.loc[:"2020-07"]  # start of taq data
print(f"Using values for {arb.columns} in the composite mispricing index.")

if flag_mean:
    arb = arb.mean(axis=1)
else:
    arb = arb.groupby("date").transform(lambda x: pd.qcut(x, q=index_q, labels=False, duplicates="drop"))
    arb = arb.sum(axis=1)


arb.name = "opt_mispricing"
arb = arb.to_frame()
arb = arb.reset_index()
arb = arb.merge(mispricing, on=["date", "permno"], how="left")
arb = arb.set_index(["date", "optionid"])


arb["opt_mispricing"] = arb.groupby("date").opt_mispricing.apply(
    lambda x: pd.qcut(x, q=pf_q, labels=False, duplicates="drop")
)

arb["combined_mispricing"] = arb[["opt_mispricing", "mispricing_score"]].mean(axis=1, skipna=True)
arb["combined_mispricing"] = arb.groupby("date").combined_mispricing.apply(
    lambda x: pd.qcut(x, q=pf_q, labels=False, duplicates="drop")
)
arb.loc["2017":, "combined_mispricing"] = np.nan


predictions = prediction_dict["N-En"]["predictions"]
predictions = predictions.merge(arb, on=["date", "optionid"], how="left")


friction_tables_and_graphs(["opt_mispricing"], "mispricing")
friction_tables_and_graphs(["combined_mispricing"], "combined_mispricing")


# %% ----------------- INSTITUTIONALS ----------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

cols_to_use = ["split_inst_share_quintile", "split_analyst_quintile"]


# %% OOS R2:
predictions = prediction_dict["N-En"]["predictions"]
predictions = predictions.merge(class_groups[["date", "optionid"] + cols_to_use], on=["date", "optionid"], how="left")
predictions = predictions.set_index("date")

r2s = []
for col in cols_to_use:
    r2 = predictions.groupby(col).apply(lambda x: TS_R2(x[["target", "predicted"]]))
    r2 = r2.loc[~r2.index.isin(["no_data", "nan"])]
    r2["Sample"] = samples[r2.index.name]
    r2.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    r2.index.name = "Portfolio"
    r2 = r2.reset_index()
    r2s.append(r2)
r2s = pd.concat(r2s).squeeze()

stars = []
for col in cols_to_use:
    print(col)
    tmp = predictions.groupby(col)[["target", "predicted"]].apply(
        lambda x: ClarkWest(x, 12, benchmark_type="zero", cw_adjust=False)
    )
    tmp = tmp.loc[~tmp.index.isin(["no_data", "nan"])]
    tmp["Sample"] = samples[tmp.index.name]
    tmp.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    tmp.index.name = "Portfolio"
    tmp = tmp.reset_index()
    tmp = tmp.set_index(["Sample", "Portfolio"])
    stars.append(obtain_stars(tmp))
stars = pd.concat(stars)


fig, ax = plt.subplots(figsize=(width, height * 0.6), sharex=True)
plots = sns.barplot(x="Sample", y="predicted", hue="Portfolio", data=r2s, palette=sns_palette(5))
ax.set_xlabel("")
ax.set_ylabel("")
ax.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
ax.set_axisbelow(True)
for i, bar in enumerate(plots.patches):
    print(bar)
    plots.annotate(
        stars.iloc[i // 2 + 5 * (i % 2)],
        (bar.get_x() + bar.get_width() / 2, r2s["predicted"].max()),
        ha="center",
        va="center",
        size=8,
        xytext=(0, 2),
        textcoords="offset points",
    )
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("../08_figures/frictions_r2.pdf")

# %% XS R2:
predictions = prediction_dict["N-En"]["predictions"]
predictions = predictions.merge(class_groups[["date", "optionid"] + cols_to_use], on=["date", "optionid"], how="left")
predictions = predictions.set_index("date")

r2s = []
for col in cols_to_use:
    r2 = predictions.groupby(col).apply(lambda x: XS_R2(x[["target", "predicted"]]))
    r2 = r2.loc[~r2.index.isin(["no_data", "nan"])]
    r2["Sample"] = samples[r2.index.name]
    r2.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    r2.index.name = "Portfolio"
    r2 = r2.reset_index()
    r2s.append(r2)
r2s = pd.concat(r2s).squeeze()

stars = []
for col in cols_to_use:
    print(col)
    tmp = predictions.groupby(col)[["target", "predicted"]].apply(
        lambda x: ClarkWest(x, 12, benchmark_type="zero", cw_adjust=False)
    )
    tmp = tmp.loc[~tmp.index.isin(["no_data", "nan"])]
    tmp["Sample"] = samples[tmp.index.name]
    tmp.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    tmp.index.name = "Portfolio"
    tmp = tmp.reset_index()
    tmp = tmp.set_index(["Sample", "Portfolio"])
    stars.append(obtain_stars(tmp))
stars = pd.concat(stars)


fig, ax = plt.subplots(figsize=(width, height * 0.6), sharex=True)
plots = sns.barplot(x="Sample", y="predicted", hue="Portfolio", data=r2s, palette=sns_palette(5))
ax.set_xlabel("")
ax.set_ylabel("")
ax.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
ax.set_axisbelow(True)
for i, bar in enumerate(plots.patches):
    print(bar)
    plots.annotate(
        stars.iloc[i // 2 + 5 * (i % 2)],
        (bar.get_x() + bar.get_width() / 2, r2s["predicted"].max()),
        ha="center",
        va="center",
        size=8,
        xytext=(0, 2),
        textcoords="offset points",
    )
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("../08_figures/frictions_xs_r2.pdf")


# %% Trading strat table:
predictions = prediction_dict["N-En"]["predictions"]
predictions = predictions.merge(class_groups[["date", "optionid"] + cols_to_use], on=["date", "optionid"], how="left")
predictions = predictions.set_index("date")

returns = {}
tstats = {}
stars = {}
n_port = 5
for col in cols_to_use:
    tmp = predictions[predictions[col] != "nan"].groupby(col).apply(ts_payoff, n_port)
    tmp = tmp.reset_index()
    tmp[col] = tmp[col].astype("f4")
    tmp = tmp.set_index([col, "date"])
    tmp["H-L"] = tmp[n_port - 1] - tmp[0]
    hml = tmp.loc[n_port - 1] - tmp.loc[0]
    hml[col] = "H-L"
    hml = hml.reset_index().set_index([col, "date"])
    tmp = pd.concat((tmp, hml))

    means = tmp.groupby(col).mean() * 100
    means.columns = ["Low Pred.", "2", "3", "4", "High Pred.", "H-L"]
    means.index = ["Low", "2", "3", "4", "High", "H-L"]
    ts = (
        tmp.groupby(col)
        .apply(
            lambda x: x.apply(
                lambda y: sm.OLS(y, np.ones(y.shape)).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues
            )
        )
        .reset_index(level=1, drop=True)
    )
    ts.columns = [r"$\text{Low} \hat{r}$", "2", "3", "4", r"$\text{High} \hat{r}$", "H-L"]
    ts.index = ["Low", "2", "3", "4", "High", "H-L"]
    star = pd.DataFrame(index=ts.index, columns=ts.index, data="")
    star[(ts.abs() > 1.65).values] = "*"
    star[(ts.abs() > 1.96).values] = "**"
    star[(ts.abs() > 2.56).values] = "***"

    returns[samples[col]] = means
    tstats[samples[col]] = ts
    stars[samples[col]] = star


def latex_table(returns, stars, save_name: str, num_format="%.4f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    first_key = list(returns.keys())[0]
    N = 1 + returns[first_key].shape[1]
    table = Tabular(
        "".join(["l"] + ["r"] * (N - 1)),
        booktabs=True,
    )
    print(table)

    to_add = [""]
    for d in returns[first_key].columns.tolist():
        if "$" in d:
            to_add.append(MultiColumn(1, align="c", data=math(d)))
        else:
            to_add.append(MultiColumn(1, align="c", data=d))
    table.add_row(to_add)
    table.add_hline()

    for i_key, key in enumerate(returns.keys()):
        table.add_row([MultiColumn(N, align="c", data=key)])
        table.add_hline()

        for i, (idx, vals) in enumerate(returns[key].iterrows()):
            to_add = [idx]
            for j, v in enumerate(vals):
                to_add.append(num_format % v + stars[key].iloc[i, j])
            table.add_row(to_add)

        if i_key != (len(returns.keys()) - 1):
            table.add_hline()
            table.add_empty_row()

    table.generate_tex("../08_figures/%s" % save_name)


file = "friction_trading"
latex_table(returns, stars, file, "%.3f")

with open("../08_figures/" + file + ".tex") as f:
    lines = f.readlines()

lines[0] = lines[0].replace(
    "l" + "r" * (n_port + 1),
    r"l!{\quad}" + r"S[table-format = 2.3,table-space-text-post={*}]!{\quad}" * (n_port + 1),
)

with open("../08_figures/" + file + ".tex", "w") as f:
    f.writelines(lines)

# %% ----------------- OPTION DEMAND -----------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

cols_to_use = ["split_joint_q_permno"]


# %% OOS R2 Heatmap:
data = prepare_model_data("N-En", cols_to_use, "2011-01-01")
data = data["predicted"]
data = data.to_frame()
data[["retail", "inst"]] = data.index.get_level_values(1).str.split("_", expand=True).tolist()
data = data.set_index(["inst", "retail"])
data = data.unstack()
data.columns = data.columns.get_level_values(1)
data = data.iloc[[0, 2, 1], [0, 2, 1]]

predictions = prediction_dict["N-En"]["predictions"]
predictions = predictions.loc["2011":]
predictions = predictions.merge(class_groups[["date", "optionid"] + cols_to_use], on=["date", "optionid"], how="left")
predictions = predictions.set_index("date")
stars = []
for col in cols_to_use:
    print(col)
    tmp = predictions.groupby(col)[["target", "predicted"]].apply(
        lambda x: ClarkWest(x, 12, benchmark_type="zero", cw_adjust=False)
    )
    tmp = tmp.drop(index=["no_data"])
    tmp.index = [samples[i] for i in tmp.index]
    if col.endswith("_inv") or "_inv_" in col:
        print("END")
        tmp.index = tmp.index.str.replace("V", "I")
    stars.append(obtain_stars(tmp))
stars = pd.concat(stars)
stars = stars.to_frame()
stars[["retail", "inst"]] = stars.index.str.split("_", expand=True).tolist()
stars = stars.set_index(["inst", "retail"])
stars = stars.unstack()
stars.columns = stars.columns.get_level_values(1)
stars = stars[data.columns]
stars = stars.loc[data.index]


annot = []
for i, col in enumerate(data.columns):
    tmp = data[col].map("{:.2%}".format) + "\n" + stars.iloc[:, i]
    annot.append(tmp)
annot = pd.concat(annot, axis=1)


fig, ax = plt.subplots(figsize=(width, height * 0.8), sharey=True)
cmap = sns_palette(100)
sns.heatmap(data, annot=annot, ax=ax, cmap=cmap, fmt="")
ax.set_yticklabels(ax.get_yticklabels(), va="center")
ax.set_ylabel("")
ax.set_xlabel("")
fig.tight_layout()
fig.savefig("../08_figures/frictions_heatmap_r2.pdf")


# %% XS R2 Heatmap:
predictions = prediction_dict["N-En"]["predictions"]
predictions = predictions.loc["2011":]


flows = option_flows_permno[["date", "permno"]]
flows["prof"] = return_option_flows(option_flows_permno, "svol", ["prop", "firm", "pcust"]) / return_option_flows(
    option_flows_permno, "vol", ["cust", "prop", "firm", "pcust"]
)
flows["prof"] = flows["prof"].replace([-np.inf, np.inf, np.nan], 0)

flows["cust"] = return_option_flows(option_flows_permno, "svol", ["cust"]) / return_option_flows(
    option_flows_permno, "vol", ["cust", "prop", "firm", "pcust"]
)
flows["cust"] = flows["cust"].replace([-np.inf, np.inf, np.nan], 0)

# merge:
predictions = predictions.merge(flows, on=["date", "permno"])
predictions = predictions.set_index("date")

predictions["prof_q"] = quantile_cuts(predictions, "prof", 3, ["Sell", "Flat", "Buy"]).values
predictions["cust_q"] = quantile_cuts(predictions, "cust", 3, ["Sell", "Flat", "Buy"]).values
predictions["split_joint_q_permno"] = predictions["prof_q"] + "_" + predictions["cust_q"]


r2 = predictions.groupby("split_joint_q_permno").apply(lambda x: XS_R2(x[["target", "predicted"]]))
r2.index = [samples[i] for i in r2.index]
r2[["prof", "cust"]] = r2.index.str.split("_", expand=True).tolist()
r2 = r2.set_index(["cust", "prof"])
r2 = r2.unstack()
r2.columns = r2.columns.get_level_values(1)
r2 = r2.iloc[[0, 2, 1], [0, 2, 1]]


stars = []
for col in cols_to_use:
    print(col)
    tmp = predictions.groupby(col)[["target", "predicted"]].apply(
        lambda x: ClarkWest(x, 12, benchmark_type="zero", cw_adjust=False)
    )
    tmp.index = [samples[i] for i in tmp.index]
    if col.endswith("_inv") or "_inv_" in col:
        print("END")
        tmp.index = tmp.index.str.replace("V", "I")
    stars.append(obtain_stars(tmp))
stars = pd.concat(stars)
stars = stars.to_frame()
stars[["retail", "inst"]] = stars.index.str.split("_", expand=True).tolist()
stars = stars.set_index(["inst", "retail"])
stars = stars.unstack()
stars.columns = stars.columns.get_level_values(1)
stars = stars[r2.columns]
stars = stars.loc[r2.index]


annot = []
for i, col in enumerate(r2.columns):
    tmp = r2[col].map("{:.2%}".format) + "\n" + stars.iloc[:, i]
    annot.append(tmp)
annot = pd.concat(annot, axis=1)


fig, ax = plt.subplots(figsize=(width, height * 0.8), sharey=True)
cmap = sns_palette(100)
sns.heatmap(r2, annot=annot, ax=ax, cmap=cmap, fmt="")
ax.set_yticklabels(ax.get_yticklabels(), va="center")
ax.set_ylabel("")
ax.set_xlabel("")
fig.tight_layout()
ax.set_yticklabels(ax.get_yticklabels(), va="center")
fig.savefig("../08_figures/frictions_heatmap_xs_r2.pdf")


# %% Trading strat option demand
cols_to_use = ["split_cust_q_permno", "split_prof_q_permno"]
predictions = prediction_dict["N-En"]["predictions"]
predictions = predictions.loc["2011":]

predictions = predictions.merge(class_groups[["date", "optionid"] + cols_to_use], on=["date", "optionid"])

returns = {}
tstats = {}
stars = {}
n_port = 5
for col in cols_to_use:
    tmp = predictions.groupby(col).apply(ts_payoff, n_port)
    tmp = tmp.drop(index=["no_data"])
    tmp["H-L"] = tmp[n_port - 1] - tmp[0]

    means = tmp.groupby(col).mean() * 100
    means.columns = ["Low Pred.", "2", "3", "4", "High Pred.", "H-L"]
    means.index = ["Sell", "Flat", "Buy"]
    ts = (
        tmp.groupby(col)
        .apply(
            lambda x: x.apply(
                lambda y: sm.OLS(y, np.ones(y.shape)).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues
            )
        )
        .reset_index(level=1, drop=True)
    )
    ts.columns = [r"$\text{Low} \hat{r}$", "2", "3", "4", r"$\text{High} \hat{r}$", "H-L"]
    ts.index = ["Sell", "Flat", "Buy"]
    star = pd.DataFrame(index=ts.index, columns=ts.columns, data="")
    star[(ts.abs() > 1.65).values] = "*"
    star[(ts.abs() > 1.96).values] = "**"
    star[(ts.abs() > 2.56).values] = "***"

    returns[samples[col.replace("_permno", "")]] = means
    tstats[samples[col.replace("_permno", "")]] = ts
    stars[samples[col.replace("_permno", "")]] = star


# %% ----------------- MISPRICING --------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# %% Do professionals pick up on mispricing in options?

# ---- get rv per permno/date for 1Q lookback window:
taq_files = glob.glob("../03_data/taq/*.pq")
taq_data = []
for file in taq_files:
    data = pd.read_parquet(file, columns=["datetime", "permno", "r2_to_close", "r2_from_prev_close"])
    data["r2"] = data["r2_to_close"] + data["r2_from_prev_close"]
    data = data.drop(columns=["r2_to_close", "r2_from_prev_close"])
    data = data.set_index("datetime")
    data.r2 *= 252  # annual rv
    taq_data.append(data)
taq_data = pd.concat(taq_data)

# rolling_rv = (
#     taq_data.groupby("permno")
#     .apply(lambda x: x.rolling("63D", min_periods=21).mean())
#     .drop(columns=["permno"])
# ).dropna()

taq_data.index = taq_data.index.to_period("M")
rolling_rv = taq_data.groupby(["permno", "datetime"]).mean()
rolling_rv = rolling_rv.groupby("permno").apply(lambda x: x.rolling(3, min_periods=3).mean())
rolling_rv = rolling_rv.dropna()
rolling_rv.index.names = ["permno", "date"]
rolling_rv = rolling_rv.reset_index()
rolling_rv["date"] = rolling_rv["date"].dt.to_timestamp()


# ---- obtain mispricing of options
atm_options = class_groups_info[class_groups_info.bucket == "short_term_atm"]
atm_options["date"] = atm_options.date.dt.to_timestamp()
atm_options["exdate"] = atm_options["date"] + pd.to_timedelta(atm_options["ttm"], "D")
atm_options["ttm"] /= 365
atm_options["C"] = atm_options["C"].replace({1: 1, 0: -1})
atm_options["K"] = atm_options["moneyness"] * atm_options["close"]

# get r
r = pd.read_csv("../03_data/yield_curve.zip", index_col=0, parse_dates=True)
r.rate /= 100
r.columns = ["days", "r"]
r.days /= 365
bdays = pd.bdate_range(r.index.min(), r.index.max())
bdays_to_append = bdays[~bdays.isin(r.index)]
for b in bdays_to_append:
    to_append = r[r.index == b - BDay()]
    to_append.index += BDay()
    r = r.append(to_append)
r = r.reset_index().sort_values(["date", "days"]).set_index("date")
r = r.groupby([r.index.to_period("M"), "days"]).mean()
r = r.reset_index(level="days")
r.index = r.index.to_timestamp()
r_mapped = atm_options.groupby("date", sort=False).apply(lambda x: map_r_rate(x, r, "exdate", "ttm"))
atm_options["r"] = r_mapped.values

# merge rv
atm_options = atm_options.merge(rolling_rv, on=["date", "permno"])

# get "theoretical" price
atm_options["theoretical_price"] = bsm(
    atm_options["close"].to_numpy(),
    atm_options["K"].to_numpy(),
    atm_options["r"].to_numpy(),
    np.sqrt(atm_options["r2"]).to_numpy(),
    atm_options["ttm"].to_numpy(),
    atm_options["C"].to_numpy(),
)

# get mispricing
atm_options["mispricing"] = np.log(atm_options["mid"] / atm_options["theoretical_price"])
mispricing = atm_options.groupby(["date", "permno"])["mispricing"].mean()
mispricing = mispricing.to_frame()
mispricing["mispricing_q5"] = quantile_cuts(mispricing, "mispricing", 5, [0, 1, 2, 3, 4]).values
mispricing = mispricing.reset_index()
mispricing["date"] = mispricing["date"].dt.to_period("M")

# ---- obtain trade direction of prof/cust
cols_to_use = ["split_cust_q_permno", "split_prof_q_permno"]
predictions = prediction_dict["N-En"]["predictions"]
predictions = predictions.loc["2011":]

predictions = predictions.merge(class_groups[["date", "optionid"] + cols_to_use], on=["date", "optionid"])
for col in cols_to_use:
    predictions = predictions[predictions[col] != "no_data"]


# ---- merge data:
tmp = mispricing[["date", "permno", "mispricing"]].copy()
tmp = tmp.set_index(["date", "permno"])
tmp = tmp.merge(predictions.groupby(["date", "permno"])[cols_to_use].first(), on=["date", "permno"])
tmp["mispricing_q3"] = quantile_cuts(tmp, "mispricing", 3, ["Under", "Fair", "Over"]).values
tmp = tmp.drop(columns=cols_to_use)

predictions = predictions[["date", "optionid", "permno"] + cols_to_use]
predictions = predictions.merge(tmp, on=["date", "permno"])

data = predictions.groupby(["date", "split_prof_q_permno", "mispricing_q3"]).size()
data /= data.groupby("date").transform("sum")
prof_data = data.groupby(["split_prof_q_permno", "mispricing_q3"]).mean().unstack()
prof_data = prof_data[["Under", "Fair", "Over"]]
prof_data = prof_data.loc[
    [
        "$V^I < 0$",
        r"$V^I \approx 0$",
        "$V^I > 0$",
    ]
]

data = predictions.groupby(["date", "split_cust_q_permno", "mispricing_q3"]).size()
data /= data.groupby("date").transform("sum")
cust_data = data.groupby(["split_cust_q_permno", "mispricing_q3"]).mean().unstack()
cust_data = cust_data[["Under", "Fair", "Over"]]
cust_data = cust_data.loc[
    [
        "$V^R < 0$",
        r"$V^R \approx 0$",
        "$V^R > 0$",
    ]
]


# ---- plot
vmin = min(cust_data.min().min(), prof_data.min().min())
vmax = max(cust_data.max().max(), prof_data.max().max())

fig, ax = plt.subplots(1, 2, figsize=(width, height * 0.8), sharex=True)
cmap = sns_palette(100)
annot = []
for i, col in enumerate(prof_data.columns):
    tmp = prof_data[col].map("{:.2%}".format)
    annot.append(tmp)
annot = pd.concat(annot, axis=1)
sns.heatmap(prof_data, annot=annot, ax=ax[0], cmap=cmap, fmt="", vmin=vmin, vmax=vmax, cbar=False)

annot = []
for i, col in enumerate(cust_data.columns):
    tmp = cust_data[col].map("{:.2%}".format)
    annot.append(tmp)
annot = pd.concat(annot, axis=1)
sns.heatmap(cust_data, annot=annot, ax=ax[1], cmap=cmap, fmt="", vmin=vmin, vmax=vmax, cbar=False)

for a in ax:
    a.set_yticklabels(a.get_yticklabels(), va="center")
    a.set_ylabel("")
    a.set_xlabel("")
ax[0].set_title("Prof")
ax[1].set_title("Cust")
fig.tight_layout()
# fig.savefig("../08_figures/mispricing_investor_type.pdf")


# %% OOS R2:
predictions = prediction_dict["N-En"]["predictions"].copy()
predictions = predictions.merge(mispricing, on=["date", "permno"])
predictions = predictions.set_index("date")
cols_to_use = ["mispricing_q5"]


r2s = []
for col in cols_to_use:
    r2 = predictions.groupby(col).apply(lambda x: TS_R2(x[["target", "predicted"]]))
    r2 = r2.loc[~r2.index.isin(["no_data", "nan"])]
    r2["Sample"] = samples[r2.index.name]
    r2.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    r2.index.name = "Portfolio"
    r2 = r2.reset_index()
    r2s.append(r2)
r2s = pd.concat(r2s).squeeze()

stars = []
for col in cols_to_use:
    print(col)
    tmp = predictions.groupby(col)[["target", "predicted"]].apply(
        lambda x: ClarkWest(x, 12, benchmark_type="zero", cw_adjust=False)
    )
    tmp = tmp.loc[~tmp.index.isin(["no_data", "nan"])]
    tmp["Sample"] = samples[tmp.index.name]
    tmp.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    tmp.index.name = "Portfolio"
    tmp = tmp.reset_index()
    tmp = tmp.set_index(["Sample", "Portfolio"])
    stars.append(obtain_stars(tmp))
stars = pd.concat(stars)


fig, ax = plt.subplots(figsize=(width, height * 0.6), sharex=True)
plots = sns.barplot(x="Sample", y="predicted", hue="Portfolio", data=r2s, palette=sns_palette(5))
ax.set_xlabel("")
ax.set_ylabel("")
ax.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
ax.set_axisbelow(True)
for i, bar in enumerate(plots.patches):
    print(bar)
    plots.annotate(
        stars.iloc[i],
        (bar.get_x() + bar.get_width() / 2, r2s["predicted"].max()),
        ha="center",
        va="center",
        size=8,
        xytext=(0, 2),
        textcoords="offset points",
    )
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("../08_figures/mispricing_r2.pdf")


# %% XS R2:
predictions = prediction_dict["N-En"]["predictions"].copy()
predictions = predictions.merge(mispricing, on=["date", "permno"])
predictions = predictions.set_index("date")
cols_to_use = ["mispricing_q5"]


r2s = []
for col in cols_to_use:
    r2 = predictions.groupby(col).apply(lambda x: XS_R2(x[["target", "predicted"]]))
    r2 = r2.loc[~r2.index.isin(["no_data", "nan"])]
    r2["Sample"] = samples[r2.index.name]
    r2.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    r2.index.name = "Portfolio"
    r2 = r2.reset_index()
    r2s.append(r2)
r2s = pd.concat(r2s).squeeze()

stars = []
for col in cols_to_use:
    print(col)
    tmp = predictions.groupby(col)[["target", "predicted"]].apply(
        lambda x: ClarkWest(x, 12, benchmark_type="zero", cw_adjust=False)
    )
    tmp = tmp.loc[~tmp.index.isin(["no_data", "nan"])]
    tmp["Sample"] = samples[tmp.index.name]
    tmp.index = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    tmp.index.name = "Portfolio"
    tmp = tmp.reset_index()
    tmp = tmp.set_index(["Sample", "Portfolio"])
    stars.append(obtain_stars(tmp))
stars = pd.concat(stars)


fig, ax = plt.subplots(figsize=(width, height * 0.6), sharex=True)
plots = sns.barplot(x="Sample", y="predicted", hue="Portfolio", data=r2s, palette=sns_palette(5))
ax.set_xlabel("")
ax.set_ylabel("")
ax.grid(ls="--", axis="y", color=(0.6, 0.6, 0.6), linewidth=0.5)
ax.set_axisbelow(True)
for i, bar in enumerate(plots.patches):
    print(bar)
    plots.annotate(
        stars.iloc[i],
        (bar.get_x() + bar.get_width() / 2, r2s["predicted"].max()),
        ha="center",
        va="center",
        size=8,
        xytext=(0, 2),
        textcoords="offset points",
    )
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("../08_figures/mispricing_xs_r2.pdf")


# %% Expected returns vs. mispricing trading strategies:
predictions = prediction_dict["N-En"]["predictions"].copy()
predictions = predictions.merge(mispricing, on=["date", "permno"])
predictions = predictions.set_index("date")
cols_to_use = ["mispricing_q5"]

returns = {}
tstats = {}
stars = {}
n_port = 5
for col in cols_to_use:
    tmp = predictions[predictions[col] != "nan"].groupby(col).apply(ts_payoff, n_port)
    tmp = tmp.reset_index()
    tmp[col] = tmp[col].astype("f4")
    tmp = tmp.set_index([col, "date"])
    tmp["H-L"] = tmp[n_port - 1] - tmp[0]
    hml = tmp.loc[n_port - 1] - tmp.loc[0]
    hml[col] = "H-L"
    hml = hml.reset_index().set_index([col, "date"])
    tmp = pd.concat((tmp, hml))

    vs_fair = []
    for n in range(n_port):
        vs_fair.append(tmp.loc[n, "H-L"] - tmp.loc[2, "H-L"])
    vs_fair = pd.concat(vs_fair, axis=1)
    vs_fair.columns = range(n_port)
    vs_fair[2] = np.nan
    vs_means = vs_fair.mean().tolist() + [np.nan]
    vs_ts = vs_fair.apply(lambda y: sm.OLS(y, np.ones(y.shape)).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues)
    vs_ts = vs_ts.T.squeeze().tolist() + [np.nan]

    means = tmp.groupby(col).mean() * 100
    means.columns = ["Low Pred.", "2", "3", "4", "High Pred.", "H-L"]
    means.index = ["Under", "2", "Fair", "4", "Over", "Over-Under"]
    means["vs. Fair"] = np.array(vs_means) * 100

    ts = (
        tmp.groupby(col)
        .apply(
            lambda x: x.apply(
                lambda y: sm.OLS(y, np.ones(y.shape)).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).tvalues
            )
        )
        .reset_index(level=1, drop=True)
    )
    ts.columns = [r"$\text{Low} \hat{r}$", "2", "3", "4", r"$\text{High} \hat{r}$", "H-L"]
    ts.index = ["Low", "2", "3", "4", "High", "H-L"]
    ts["vs. Fair"] = vs_ts

    star = pd.DataFrame(index=ts.index, columns=ts.columns, data="")
    star[(ts.abs() > 1.65).values] = "*"
    star[(ts.abs() > 1.96).values] = "**"
    star[(ts.abs() > 2.56).values] = "***"

    returns[samples[col]] = means
    tstats[samples[col]] = ts
    stars[samples[col]] = star


def latex_table(returns, stars, save_name: str, num_format="%.4f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    first_key = list(returns.keys())[0]
    N = 1 + returns[first_key].shape[1]
    table = Tabular(
        "".join(["l"] + ["r"] * (N - 1)),
        booktabs=True,
    )
    print(table)

    to_add = [""]
    for d in returns[first_key].columns.tolist():
        if "$" in d:
            to_add.append(MultiColumn(1, align="c", data=math(d)))
        else:
            to_add.append(MultiColumn(1, align="c", data=d))
    table.add_row(to_add)
    table.add_hline()

    for i_key, key in enumerate(returns.keys()):
        table.add_row([MultiColumn(N, align="c", data=key)])
        table.add_hline()

        for i, (idx, vals) in enumerate(returns[key].iterrows()):
            to_add = [idx]
            for j, v in enumerate(vals):
                if np.isnan(v):
                    to_add.append("---")
                else:
                    to_add.append(num_format % v + stars[key].iloc[i, j])
            table.add_row(to_add)

        if i_key != (len(returns.keys()) - 1):
            table.add_hline()
            table.add_empty_row()

    table.generate_tex("../08_figures/%s" % save_name)


file = "mispricing_trading"
latex_table(returns, stars, file, "%.3f")

with open("../08_figures/" + file + ".tex") as f:
    lines = f.readlines()

lines[0] = lines[0].replace(
    "l" + "r" * (n_port + 1),
    r"l!{\quad}" + r"S[table-format = 2.3,table-space-text-post={*}]!{\quad}" * (n_port + 2),
)

with open("../08_figures/" + file + ".tex", "w") as f:
    f.writelines(lines)