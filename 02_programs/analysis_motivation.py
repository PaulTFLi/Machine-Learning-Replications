# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

import pandas as pd
import glob
from joblib import Parallel, delayed

cm = 1 / 2.54
width = 15.92 * cm
height = 10 * cm

import matplotlib.pyplot as plt


option_data = []
files = glob.glob("../04_results/option_sample/classifier_*.pq")
for file in files:
    print(file)
    tmp = pd.read_parquet(file, columns=["date", "permno", "optionid", "iv", "doi", "corr", "bucket", "baspread"])
    option_data.append(tmp)
option_data = pd.concat(option_data)
option_data.date = option_data.date.dt.to_timestamp()
option_data.optionid = option_data.optionid.astype(int)

characteristics_to_use = ["idiovol", "dy", "ivslope", "ivvol", "volunc", "civpiv", "roeq", "roaq", "defrisk"]
char_files = glob.glob("../04_results/option_sample/characteristics*.pq")


def read_files(file):
    data = pd.read_parquet(file, columns=characteristics_to_use + ["optionid"])
    data = data.set_index("optionid", append=True)

    return data


data = Parallel(n_jobs=4, verbose=True)(delayed(read_files)(file) for file in char_files)
data = pd.concat(data)
data = data.reset_index()

option_data = pd.merge(option_data, data, on=["date", "optionid"])

# %%% Time series correlation of iv and bid-ask spread
iv = option_data.groupby("date").apply(lambda x: (x.iv * x.doi).sum() / x.doi.sum())
iv.name = "iv"
idiovol = option_data.groupby("date").apply(lambda x: (x.idiovol * x.doi).sum() / x.doi.sum())
idiovol.name = "idiovol"

iv_corr = pd.merge(iv, idiovol, on="date")
print(iv_corr.corr())
iv_corr = iv_corr.rolling(60).corr(pairwise=True)
iv_corr = iv_corr.loc[iv_corr.index.get_level_values(1) == "idiovol", "iv"]
iv_corr = iv_corr.dropna()
iv_corr = iv_corr.reset_index(level=1, drop=True)

fig, ax = plt.subplots(1, 1, figsize=(width, height * 0.75), dpi=800, sharex=True)
ax.plot(iv_corr, ls="-", marker=None, c="k", linewidth=1)
ax.margins(x=0)
ax.set_ylim([-1, 1])
ax.hlines(y=0, xmin=iv_corr.index.min(), xmax=iv_corr.index.max(), color="k", ls="--", linewidth=0.5)
fig.tight_layout()
fig.savefig("../08_figures/corr_iv_idiovol.pdf")
