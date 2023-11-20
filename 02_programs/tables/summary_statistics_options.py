# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""
import pandas as pd
import numpy as np
import datetime
import glob

from pylatex import (
    Tabular,
    MultiColumn,
    NoEscape,
    Command,
    LongTable,
)
from scipy.stats import skew, kurtosis, jarque_bera

def bucket_stats(df, index=0, jb = True):
    if jb:
        jb = jarque_bera(df)[1]
    else:
        jb = np.nan
    df_w = df[(df > np.percentile(df,0.1)) & (df < np.percentile(df,99.9))]
    return pd.DataFrame(
        {
            "Mean": np.nanmean(df),
            "Sd": np.nanstd(df),
            "10-Pctl": np.nanpercentile(df, 10),
            "Q1": np.nanpercentile(df, 25),
            "Q2": np.nanmedian(df),
            "Q3": np.nanpercentile(df, 75),
            "90-Pctl": np.nanpercentile(df, 90),
            "Skew": skew(df_w),
            "Kurt": kurtosis(df_w, fisher = True),
            "JB": jb
        },
        index=[index],
    )


def return_stats(df, index=0, jb = True):
    if jb:
        jb = jarque_bera(df)[1]
    else:
        jb = np.nan
    df_w = df[(df > np.percentile(df,0.1)) & (df < np.percentile(df,99.9))]

    return pd.DataFrame(
        {
            "Mean": np.nanmean(df),
            "Sd": np.nanstd(df),
            "10-Pctl": np.nanpercentile(df, 10),
            "Q1": np.nanpercentile(df, 25),
            "Q2": np.nanmedian(df),
            "Q3": np.nanpercentile(df, 75),
            "90-Pctl": np.nanpercentile(df, 90),
            "Skew": skew(df_w),
            "Kurt": kurtosis(df_w, fisher= True),
            "JB": jb
        },
        index=[index],
    )


def long_short_term_classification(inp):
    if "short_term" in inp:
        return "short_term"
    else:
        return "long_term"


def pos_to_char(pos):
    return chr(pos + 97)


# %% Return summary stats
index_names = {
    "return_dh_daily_inv": "Delta-Hedged Return",
    "ttm": "Days to Maturity",
    "moneyness": "Moneyness",
    "iv": "Implied volatility",
    "delta": "Absolute Delta",
}

files = glob.glob("../../04_results/option_sample/classifier*.pq")
returns = pd.concat([pd.read_parquet(file,
                                     columns = ["date","optionid","bucket",
                                                "return_dh_daily_inv",
                                                "ttm",
                                                "moneyness",
                                                "iv",
                                                "delta",
                                                "C",
                                                "P",
                                                "permno"]) for file in files])
returns.date = pd.to_datetime(returns.date.astype(str))

n_optionid = len(returns.optionid.unique())
n_stocks = len(returns.permno.unique())

returns.return_dh_daily_inv *= 100
returns.iv *= 100
returns.delta = returns.delta.abs()
calls = returns.loc[returns.C == 1, :]
puts = returns.loc[returns.P == 1, :]
pre2003 = returns.loc[returns.date < datetime.datetime(2003, 1, 1), :]
post2003 = returns.loc[returns.date >= datetime.datetime(2003, 1, 1), :]

columns = ["return_dh_daily_inv", "ttm", "moneyness", "iv", "delta"]
panel_data = []
for data in [returns, calls, puts, pre2003, post2003]:
    tmp = []
    for col in columns:
        if col == "return_dh_daily_inv":
            tmp.append(return_stats(data[col], col, True))
        else:
            tmp.append(return_stats(data[col], col, False))
    tmp = pd.concat(tmp)
    tmp = tmp.round(2)
    tmp.rename(index_names, axis=0, inplace=True)
    panel_data.append(tmp)

nbr_all = returns.shape[0]
nbr_calls = calls.shape[0]
nbr_puts = puts.shape[0]
nbr_pre = pre2003.shape[0]
nbr_post = post2003.shape[0]

panel_a_name = "Panel A: All Options (N=" + "{:,}".format(nbr_all) + ")"
panel_b_name = "Panel B: Call Options (N=" + "{:,}".format(nbr_calls) + ")"
panel_c_name = "Panel C: Put Options (N=" + "{:,}".format(nbr_puts) + ")"
panel_d_name = "Panel D: All Options 1996-2002 (N=" + "{:,}".format(nbr_pre) + ")"
panel_e_name = "Panel E: All Options 2003-2020 (N=" + "{:,}".format(nbr_post) + ")"

panels = zip(panel_data, [panel_a_name, panel_b_name, panel_c_name, panel_d_name, panel_e_name])

ncol = panel_data[0].shape[1] + 1
tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
column_names = [""]
column_names.extend(list(panel_data[0].columns.values))
tabular.add_row(column_names)

for panel, name in panels:

    tabular.add_hline(1, ncol)
    tabular.add_row((MultiColumn(ncol, align="l", data=name),))
    tabular.add_hline(1, ncol)

    for index, row in panel.iterrows():
        row = row
        cells = [index.replace("_", " ")]
        for r in row:
            if np.isnan(r):
                r = ""
            cells.append(r)
        tabular.add_row(cells)

tabular.generate_tex("../../08_figures/summary_returns_addon")

# %% Bucket overview table -- RETURNS

short_term = returns.loc[returns.bucket.apply(lambda x: "short_term" in x), :].copy()
long_term = returns.loc[returns.bucket.apply(lambda x: "long_term" in x), :].copy()
short_term["bucket"] = "short_term"
long_term["bucket"] = "long_term"
returns = pd.concat([returns, short_term, long_term])

panel_data = []
panel_names = []
buckets = np.sort(returns.bucket.unique())
i = 0
for bucket in buckets:
    tmp = []
    data_tmp = returns.loc[returns.bucket == bucket, :]
    for col in columns:
        if col == "return_dh_daily_inv":
            tmp.append(return_stats(data_tmp[col], col, True))
        else:
            tmp.append(return_stats(data_tmp[col], col, False))
    tmp = pd.concat(tmp)
    tmp = tmp.round(2)
    tmp.rename(index_names, axis=0, inplace=True)
    panel_data.append(tmp)
    panel_name = (
        "Panel "
        + pos_to_char(i).upper()
        + ": "
        + bucket.replace("_", " ").title()
        + " (N="
        + "{:,}".format(data_tmp.shape[0])
        + ")"
    )
    panel_names.append(panel_name)
    i += 1
    print(bucket)

panels = list(zip(panel_data, panel_names))
pagebreaks = [5, 10]
cols = list(panel_data[0].columns.values)
cols.insert(0, "")
ncol = len(cols)

label = "tab:summaryreturnsbuckets"
caption = "captionsummaryreturnsbuckets"
filename = "../../08_figures/summary_returns_buckets_addon.tex"

tabular = LongTable("l" + "c" * (ncol - 1), booktabs=True)
tabular.add_row(cols)
tabular.add_hline()
tabular.append(Command("endfirsthead"))

tabular.add_row((MultiColumn(ncol, align="l", data=NoEscape("Table \\thetable \ from previous page")),))
tabular.add_hline()
tabular.add_row(cols)
tabular.end_table_header()

tabular.add_hline()
tabular.add_row((MultiColumn(ncol, align="r", data="Continued on Next Page"),))
tabular.add_hline()
tabular.end_table_footer()

# tabular.add_hline()
tabular.add_row((MultiColumn(ncol, align="r", data="Not Continued on Next Page"),))
tabular.add_hline()
tabular.end_table_last_footer()

i = 0
for panel, name in panels:
    if i > 0:
        tabular.add_hline(1, ncol)
    tabular.add_row((MultiColumn(ncol, align="l", data=name),))
    tabular.add_hline(1, ncol)
    for index, row in panel.iterrows():
        row = row
        cells = [index.replace("_", " ")]
        for r in row:
            if np.isnan(r):
                r = ""
            cells.append(r)
        tabular.add_row(cells)
    if (i + 1) in pagebreaks:
        tabular.append(Command("pagebreak"))
    i += 1

tabular.generate_tex(filename.replace(".tex", ""))


f = open(filename)
lines = f.readlines()
f.close()

# insert caption and label
idx_tabular = np.argwhere(["\\toprule" in line for line in lines]).T[0][0]
caption_label = "\\caption{\\" + caption + "} \\label{" + label + "} \\\\%\n"
lines.insert(idx_tabular, caption_label)

with open(filename, mode="w") as f:
    f.write("% generated by python\n")
f.close()

with open(filename, mode="a") as f:
    for line in lines:
        f.write(line)
f.close()


# %% Bucket overview table -- Count
permnos = returns[["date", "permno", "bucket"]]
counts = permnos.groupby(["date", "permno", "bucket"]).apply(lambda x: x.shape[0])
counts = pd.DataFrame(counts)
counts.reset_index(inplace=True)
counts = counts.groupby(["date", "bucket"]).apply(lambda x: bucket_stats(x[0]))
counts.reset_index(inplace=True)
counts.drop(columns=["level_2", "date"], inplace=True)

counts = counts.groupby(["bucket"]).mean()
counts = counts.round(2)

ncol = counts.shape[1] + 1
tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
column_names = [""]
column_names.extend(list(counts.columns.values))
tabular.add_row(column_names)
tabular.add_hline(1, ncol)

for index, row in counts.iterrows():
    row = row
    cells = [index.replace("_", " ").title()]
    for r in row:
        cells.append(r)
    tabular.add_row(cells)

tabular.generate_tex("../../08_figures/summary_bucket_options")