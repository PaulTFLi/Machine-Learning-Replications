# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

from glob import glob
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import seaborn as sns
import statsmodels.api as sm

from ipca import InstrumentedPCA

import matplotlib.pyplot as plt


def formatter(inp):
    if isinstance(inp, str):
        if inp.lstrip("-").replace(".", "", 1).isdigit():
            inp = float(inp)
        else:
            inp = np.nan
    return inp


def capm(x):
    mod = sm.OLS(x["xretx"], sm.add_constant(x["xvwretx"]), missing="drop").fit()
    return mod.params


# %%% settings

oos_years = range(2003, 2021)

# # top 50 characteristics from univariate portfolio sorts
characteristics_to_use = [
    "gamma",
    "mve",
    "iv",
    "ill",
    "atm_iv",
    "dolvol",
    "rd_sale",
    "embedlev",
    "pzeros",
    "rnk182",
    "ailliq",
    "moneyness",
    "rnk273",
    "volga",
    "close",
    "ivarud30",
    "dvol",
    "idiovol",
    "stdacc",
    "roaq",
    "stdcf",
    "roic",
    "rnk365",
    "pfht",
    "ivrv",
    "baspread",
    "cashdebt",
    "rnk91",
    "std_dolvol",
    "theta",
    "volunc",
    "roavol",
    "rd_mve",
    "ep",
    "vol",
    "rv",
    "retvol",
    "iv_rank",
    "maxret",
    "ivd",
    "skewiv",
    "cfp",
    "roeq",
    "std_turn",
    "optspread",
    "rns91",
    "rnk30",
    "operprof",
    "ivrv_ratio",
    "pba",
]

# %%% Load data
char_files = glob("../04_results/option_sample/characteristics*.pq")
classifier = glob("../04_results/option_sample/classifier*.pq")

target = pd.concat(
    [pd.read_parquet(file, columns=["date", "permno", "optionid", "doi", "return_dh_daily_inv"]) for file in classifier]
)

target["date"] = target["date"].dt.to_timestamp()
target = target.set_index(["date", "optionid"]).sort_index()
target = target.rename(columns={"doi": "weight"})


def read_files(file):
    data = pd.read_parquet(file, columns=characteristics_to_use + ["optionid"])
    data = data.set_index("optionid", append=True)

    # create ranks
    return data


data = Parallel(n_jobs=cpu_count(), verbose=True)(delayed(read_files)(file) for file in char_files)
data = pd.concat(data)

# add target variable + weighting
data = data.merge(target, on=["date", "optionid"])

# conform with expected input into IPCA
data = data.reset_index().set_index(["optionid", "date"])

crsp_market = pd.read_csv("../03_data/crsp_market_returns.csv")
crsp_market["DATE"] = pd.to_datetime(crsp_market.DATE, format="%Y%m%d")
crsp_market.columns = [c.lower() for c in crsp_market.columns]
crsp_market.vwretx = crsp_market.vwretx.apply(lambda x: formatter(x))


data = data.reset_index()
vwret = crsp_market[["date", "vwretx"]]
vwret.date = vwret.date.dt.to_period("M")
vwret.date = vwret.date.dt.to_timestamp()
data = data.merge(vwret[["date", "vwretx"]], on="date")
data = data.set_index(["optionid", "date"])

# %%% IPCA with two exog. factors, delta-hedged returns and all options
ipca_option_returns_dh = []
for oos_year in oos_years:
    print("Fit IPCA for OOS year = {}".format(oos_year))
    data_is = data.loc[data.index.get_level_values(1).year < oos_year, :]
    data_oos = data.loc[data.index.get_level_values(1).year == oos_year, :]

    y_is = data_is[["return_dh_daily_inv"]]
    x_is = data_is[characteristics_to_use]

    vw_market_return_is = data_is.groupby(level=1).apply(
        lambda x: (x.return_dh_daily_inv * x.weight).sum() / x.weight.sum()
    )
    vw_market_return_is.name = "vwretx_o"
    vw_smarket_return_is = data_is.reset_index()[["date", "vwretx"]].drop_duplicates()
    vw_smarket_return_is = vw_smarket_return_is.sort_values("date").set_index("date")
    vw_market_returns = pd.merge(vw_market_return_is, vw_smarket_return_is, left_index=True, right_index=True)

    psf = np.array(vw_market_returns).T

    regr = InstrumentedPCA(n_factors=2, intercept=True)
    regr = regr.fit(X=x_is, y=y_is, PSF=psf)
    Gamma, Factors = regr.get_factors(label_ind=True)

    for m in np.sort(data_oos.index.get_level_values(1).month.unique()):
        tmp = data_oos.loc[data_oos.index.get_level_values(1).month == m, :]
        vw_ret = vw_market_return_is.mean()
        vw_ret_o = vw_smarket_return_is.mean().iloc[0]
        x_os = tmp[characteristics_to_use]

        expected_ret = (
            np.matmul(np.array(x_os), np.array(Gamma[0])) * vw_ret
            + np.matmul(np.array(x_os), np.array(Gamma[1])) * vw_ret_o
        )
        expected_ret = expected_ret + np.matmul(np.array(x_os), np.array(Gamma[2]))
        expected_ret = pd.DataFrame(expected_ret, columns=["e_ret_o"], index=x_os.index)
        expected_ret = expected_ret.merge(tmp[["return_dh_daily_inv"]], left_index=True, right_index=True)
        ipca_option_returns_dh.append(expected_ret)

ipca_option_returns_dh = pd.concat(ipca_option_returns_dh)
ipca_option_returns_dh = ipca_option_returns_dh.reset_index()
ipca_option_returns_dh.groupby("date").e_ret_o.median().plot()

# %%% Load N-En and L-En Predictions

from analysis_setup import prediction_dict, analysisLoc, width, height, sns_palette
import os

class_groups = pd.read_parquet(os.path.join(analysisLoc, "class_groups.pq"))
class_groups_with_info = pd.read_parquet(
    os.path.join(analysisLoc, "class_groups_and_info.pq"),
    columns=[
        "date",
        "optionid",
        "doi",
    ],
)

class_groups_with_info.date = class_groups_with_info.date.dt.to_timestamp()

n_en = prediction_dict["N-En"]["predictions"].copy()
n_en = n_en.rename(columns={"predicted": "N-En"})
n_en = n_en.reset_index()
n_en.date = n_en.date.dt.to_timestamp()
l_en = prediction_dict["L-En"]["predictions"].copy()
l_en = l_en.rename(columns={"predicted": "L-En"})
l_en = l_en.reset_index()
l_en.date = l_en.date.dt.to_timestamp()

ts_data = pd.merge(
    n_en[["date", "optionid", "N-En"]], l_en[["date", "optionid", "L-En", "target"]], on=["date", "optionid"]
)
ts_data = ts_data.merge(ipca_option_returns_dh[["date", "optionid", "e_ret_o"]], on=["date", "optionid"])
ts_data = ts_data.merge(class_groups_with_info, on=["date", "optionid"])

output = []
for col in ["N-En", "L-En", "e_ret_o"]:
    tmp = ts_data[["date", "optionid", "doi", col, "target"]].copy()

    tmp["port"] = tmp.groupby(["date"])[col].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))

    tmp = tmp.groupby(["date", "port"]).apply(lambda x: (x.doi * x.target).sum() / x.doi.sum())
    returns = tmp.unstack()
    returns.columns = "Lo 2 3 4 5 6 7 8 9 Hi".split()
    returns["H-L"] = returns["Hi"] - returns["Lo"]
    returns = returns.stack()
    returns.name = "return"
    returns = returns.to_frame()
    returns["type"] = col
    returns.index.names = ["date", "port"]
    output.append(returns)
output = pd.concat(output, axis=0)
output = output[output.index.get_level_values("port") == "H-L"]
output = output.reset_index(level="port", drop=True)
output["type"] = output["type"].replace({"e_ret_o": "Benchmark"})

means = output.groupby([pd.Grouper(freq="A"), "type"]).mean()
means = means.reset_index()
means.date = means.date.dt.year
means = means.rename(columns={"return": "Returns"})

fig, ax = plt.subplots(1, 1, figsize=(width, height * 0.5), dpi=800, sharex=True)
sns.barplot(data=means, x="date", y="Returns", hue="type", ax=ax, palette=sns_palette(3))
ax.legend(frameon=False, ncol=3)
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.grid(axis="y", ls="--", lw=0.5, color="k")
ax.set_axisbelow(True)
ax.grid(axis="y", ls="--", lw=0.5, color="k")
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("../08_figures/ts_over_time_with_benchmark.pdf")
