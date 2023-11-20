# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
import pandas as pd
import glob
import os
from init_model_parameters import (
    possible_samples,
)


# %% hyperparameters
print("Possible samples:")
for i, elem in enumerate(possible_samples):
    print("%d: %s" % (i, elem))

fileLoc = input("Which file location?   ")
fileLoc = possible_samples[int(fileLoc)]

classification_files = glob.glob(os.path.join(fileLoc, "classifier*"))

target_columns = [
    "date",
    "permno",
    "optionid",
]

return_column = "return_dh_daily_inv"
option_level_columns = [return_column, "ttm", "moneyness", "iv", "embedlev"]
stock_level_columns = ["ivrv"]
target_columns += option_level_columns + stock_level_columns
stock_level_columns += ["N_by_permno"]


# %%
N = 0
total_agg = []
returns = []
for file in classification_files:
    print("Working on file {}.".format(file))
    data = pd.read_parquet(file, columns=target_columns)
    returns.append(data[return_column])

    # aggregations:
    N += len(data)

    N_by_permno = data.groupby(["date", "permno"]).size()
    N_by_permno.name = "N_by_permno"
    data = data.merge(N_by_permno, on=["date", "permno"])

    # option-level aggregations:
    means = data.groupby("date")[option_level_columns].mean()
    means["type"] = "Mean"
    means = means.set_index("type", append=True)
    sds = data.groupby("date")[option_level_columns].std()
    sds["type"] = "Std"
    sds = sds.set_index("type", append=True)
    quantiles = []
    for quant in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        tmp = data.groupby("date")[option_level_columns].quantile(quant)
        if quant == 0.01:
            tmp["type"] = "%dst" % int(quant * 100)
        elif quant == 0.5:
            tmp["type"] = "Median"
        else:
            tmp["type"] = "%dth" % int(quant * 100)
        tmp = tmp.set_index("type", append=True)
        quantiles.append(tmp)
    quantiles = pd.concat(quantiles)
    option_agg = pd.concat((means, sds, quantiles))

    # stock-level aggregations:
    data = data.groupby(["date", "permno"]).first()
    means = data.groupby("date")[stock_level_columns].mean()
    means["type"] = "Mean"
    means = means.set_index("type", append=True)
    sds = data.groupby("date")[stock_level_columns].std()
    sds["type"] = "Std"
    sds = sds.set_index("type", append=True)
    quantiles = []
    for quant in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        tmp = data.groupby("date")[stock_level_columns].quantile(quant)
        if quant == 0.01:
            tmp["type"] = "%dst" % int(quant * 100)
        elif quant == 0.5:
            tmp["type"] = "Median"
        else:
            tmp["type"] = "%dth" % int(quant * 100)
        tmp = tmp.set_index("type", append=True)
        quantiles.append(tmp)
    quantiles = pd.concat(quantiles)
    stock_agg = pd.concat((means, sds, quantiles))

    total_agg.append(pd.concat((option_agg, stock_agg), axis=1))


# %% Create summary table:
total_agg = pd.concat(total_agg)
total_agg = total_agg.groupby("type").mean().T
total_agg["N"] = N
total_agg = total_agg[["N", "Mean", "Std", "1st", "5th", "25th", "Median", "75th", "95th", "99th"]]

total_agg = total_agg.rename(
    index={
        "return_dh_daily_inv": "$r$",
        "embedlev": "embedded leverage",
        "ivrv": "$iv - rv$",
        "N_by_permno": "N options",
    }
)
total_agg.to_excel("../08_figures/summary_statistics.xlsx", engine="openpyxl")


# %% Create return histogram:
returns = pd.concat(returns)
returns.hist(bins=250)


# %%
