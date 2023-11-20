# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import datetime
from scipy import stats as ss
from get_ff12_industry import get_ff12_industry
import glob

def frequency_sectors(df):
    fr = df.groupby(["sic_name"]).apply(lambda x: x.shape[0])
    fr = fr / fr.sum()
    return fr


from pylatex import (
    Tabular,
    MultiColumn,
    Table,
)

def monthly_aggregated(df):
    vol = df.ret.std() * np.sqrt(252) * 100
    firmsize = df.mcap.iloc[0] / 10 ** 6
    traded_amex_nyse = df.nyse_amex_flag.iloc[0]

    return pd.DataFrame(
        {
            "firmsize": firmsize,
            "vol": vol,
            "nyse_amex_flag": traded_amex_nyse,
            "permno": df.permno.iloc[0],
            "ym": df.ym.iloc[0],
            "sic": df.siccd.iloc[0],
            "shrcd": df.shrcd.iloc[0],
            "exchcd": df.exchcd.iloc[0]
        },
        index=[0],
    )


def monthly_cross_section_stats(df, fld):
    val = df[fld]
    return pd.DataFrame(
        {
            "Mean": np.nanmean(val),
            "Std": np.nanstd(val),
            "10-Pctl": np.nanpercentile(val, 10),
            "Q1": np.nanpercentile(val, 25),
            "Median": np.nanmedian(val),
            "Q3": np.nanpercentile(val, 75),
            "90-Pctl": np.nanpercentile(val, 90),
        },
        index=[fld],
    )


def panel_a_stats(df):
    df2 = df.loc[df.flag_sample == 1,:].copy()
    nbr = df2.shape[0]
    ew = df.flag_sample.sum()
    vw = df2.firmsize.sum() / df.firmsize.sum() * 100
    nyse_amex = np.sum(df2.nyse_amex_flag) / nbr * 100
    prev_month = df2["ym_prev_flag"].mean() * 100

    return pd.DataFrame(
        {"Number": nbr, "EW": ew, "VW": vw, "NYSE AMEX": nyse_amex, "Prev Month": prev_month},
        index=[df.ym.iloc[0]],
    )


# %% preapre data

files = glob.glob("../../04_results/option_sample/classifier*.pq")
permnos = pd.concat([pd.read_parquet(file) for file in files])
permnos = permnos[["date","permno"]]
permnos.rename({"date": "ym"}, axis = 1, inplace = True)
permnos["flag_sample"] = 1
permnos = permnos.drop_duplicates()

crsp = pd.read_parquet(
    "../../03_data/crsp_prices_daily.pq",
    columns=["permno", "date", "close", "shrout", "primexch", "dlret", "ret", "vol", "siccd","shrcd","exchcd"],
)
crsp["shrcd"] = crsp.shrcd.astype(int)
crsp["siccd"] = crsp.siccd.astype(int)
crsp["mcap"] = crsp.close.abs() * crsp.shrout * 1000
crsp["nyse_amex_flag"] = crsp.primexch.apply(lambda x: (x == "N") | (x == "A"))

crsp = crsp.loc[crsp.date >= datetime.datetime(1995,1,1),:]

crsp = crsp.loc[(crsp.shrcd == 10) | (crsp.shrcd == 11) , :]  
crsp = crsp[crsp.exchcd.isin([1, 2, 3, 31, 32, 33])]

insti = pd.read_parquet("../../03_data/institutional_holdings_13f.pq")
insti["ym"] = insti.date.dt.to_period("M")
insti.drop(columns="date", inplace=True)

# %% add crsp market cap
crsp_mcap = crsp.groupby("date").agg({"mcap": "sum"})
crsp_mcap.rename({"mcap": "crsp_mcap"}, axis=1, inplace=True)

crsp = pd.merge(crsp, crsp_mcap, left_on="date", right_index=True)
crsp["ym"] = crsp.date.dt.to_period("M")
crsp = crsp.sort_values("date")

stats = pd.concat(
    Parallel(-4, verbose=True)(
        delayed(monthly_aggregated)(x) for _, x in crsp.groupby(["permno", "ym"])
    )
)
stats.reset_index(drop=True, inplace=True)

stats = stats.sort_values("ym")

# %% - merge with sample
stats = stats.loc[stats.ym >= permnos.ym.min(), :]
stats = stats.loc[stats.ym <= permnos.ym.max(), :]

stats.ym = stats.ym.apply(lambda x: x.to_timestamp())
stats.ym = stats.ym.dt.to_period("M")
stats = pd.merge(stats, permnos, left_on=["permno", "ym"], right_on=["permno", "ym"], how="left")
stats.flag_sample = stats.flag_sample.fillna(0)

# %% - add analysis on prev month included flag
ym = np.sort(stats.ym.unique())
stats_prev_month = []
i = 0
for idx, df in stats[["permno", "ym", "flag_sample"]].groupby("ym", as_index=False):
    df = df.loc[df.flag_sample == 1, :].copy()
    if i == 0:
        df_old = df.copy()
    else:
        permnos_to_keep = df.permno[df.permno.isin(df_old.permno)]
        stats_prev_month.append(pd.DataFrame({"permno": permnos_to_keep, "ym": idx}))
        df_old = df.copy()
    i += 1
stats_prev_month = pd.concat(stats_prev_month)
stats_prev_month["ym_prev_flag"] = 1
stats_prev_month.ym = stats_prev_month.ym.apply(lambda x: x.to_timestamp()).dt.to_period("M")

stats = pd.merge(
    stats, stats_prev_month, left_on=["permno", "ym"], right_on=["permno", "ym"], how="left"
)
stats.ym_prev_flag = stats.ym_prev_flag.fillna(0)

# %% Panel A
index_names = {
    "Number": "Number of stocks in the sample each month",
    "EW": "Stock coverage of stock universe (EW)",
    "VW": "Stock coverage of stock universe (VW)",
    "NYSE AMEX": "Stock traded at NYSE or AMEX",
    "Prev Month": "Stock already included in previous month",
}

panel_a_raw = stats.groupby("ym").apply(lambda x: panel_a_stats(x))
panel_a = []
flds = ["Number", "EW", "VW", "NYSE AMEX", "Prev Month"]
for fld in flds:
    panel_a.append(monthly_cross_section_stats(panel_a_raw, fld))
panel_a = pd.concat(panel_a)
panel_a = panel_a.round(2)
panel_a.rename(index_names, axis=0, inplace=True)

# %% Panel B
percentile_data = []
for idx, df in stats.groupby("ym"):
    df["size_percentile"] = df.firmsize.apply(lambda x: ss.percentileofscore(df.firmsize, x))
    df["vol_percentile"] = df.vol.apply(lambda x: ss.percentileofscore(df.vol, x))
    percentile_data.append(df[["permno", "ym", "size_percentile", "vol_percentile"]].copy())
percentile_data = pd.concat(percentile_data)

stats = stats.merge(insti, left_on=["ym", "permno"], right_on=["ym", "permno"], how="left")
stats = stats.merge(percentile_data, left_on=["permno", "ym"], right_on=["permno", "ym"])
panel_b_raw = stats.loc[stats.flag_sample == 1, :].copy()
panel_b_raw = panel_b_raw[["firmsize", "ym", "size_percentile", "vol_percentile"]]

panel_b = []
flds = ["firmsize", "size_percentile", "vol_percentile"]
for fld in flds:
    tmp = panel_b_raw.groupby("ym").apply(lambda x: monthly_cross_section_stats(x, fld))
    tmp.reset_index(inplace=True)
    panel_b.append(tmp)
panel_b = pd.concat(panel_b)
panel_b.drop(columns=["ym"], inplace=True)
panel_b = panel_b.groupby("level_1").mean()
panel_b = panel_b.round(0).astype(int)

index_names = {
    "firmsize": "Firm size in million",
    "size_percentile": "Firm size CSRP percentile",
    "vol_percentile": "Firm volatility CSRP percentile",
}

panel_b.rename(index_names, axis=0, inplace=True)

# %% Panel C
stats["sic_name"] = stats.sic.apply(lambda x: get_ff12_industry(x))
wth_options = stats.loc[stats.flag_sample == 1, :].copy()
wth_options = wth_options.groupby("ym").apply(lambda x: frequency_sectors(x)).mean()
crsp_uni = stats.loc[stats.flag_sample == 0, :].copy()
crsp_uni = crsp_uni.groupby("ym").apply(lambda x: frequency_sectors(x)).mean()

panel_c = pd.concat(
    [pd.DataFrame(wth_options, columns=["Options"]), pd.DataFrame(crsp_uni, columns=["CRSP"])],
    axis=1,
)

panel_c_1 = panel_c.loc[
    [
        "Consumer nondurables",
        "Consumer durables",
        "Manufacturing",
        "Energy",
        "Chemicals",
        "Business Equipment",
    ]
]
panel_c_1 = (panel_c_1 * 100).round(2).astype(str) + "%"
panel_c_1.reset_index(inplace=True)

panel_c_2 = panel_c.loc[["Telecom", "Utilities", "Wholesale", "Healthcare", "Finance", "Other"]]
panel_c_2 = (panel_c_2 * 100).round(2).astype(str) + "%"
panel_c_2.reset_index(inplace=True)

header_change_c = {
    "sic_name": "FF-12 Industry",
    "Options": "Optionable Stocks",
    "CRSP": "CRSP sample",
}

panel_c_1.rename(header_change_c, axis=1, inplace=True)

panel_c_2.rename(header_change_c, axis=1, inplace=True)

# %% build table

# Panel A
ncol = panel_a.shape[1] + 1
tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)
column_names = [""]
column_names.extend(list(panel_a.columns.values))
tabular.add_row(column_names)
tabular.add_hline(1, ncol)
tabular.add_row((MultiColumn(ncol, align="l", data="Panel A: Time-Series Distribution"),))
tabular.add_hline(1, ncol)

for index, row in panel_a.iterrows():
    row = row
    cells = [index]
    for r in row:
        cells.append(r)
    tabular.add_row(cells)

# Panel B
tabular.add_hline(1, ncol)
tabular.add_row(
    (
        MultiColumn(
            ncol, align="l", data="Panel B: Time-Series Average of Cross-Sectional Distributions"
        ),
    )
)
tabular.add_hline(1, ncol)
for index, row in panel_b.iterrows():
    row = row
    cells = [index]
    for r in row:
        cells.append(r)
    tabular.add_row(cells)
tabular.add_hline(1, ncol)

# Panel C - new tabular x
ncol = panel_c_2.shape[1] * 2
nc = int(0.5 * (ncol - 2))
header_c = list(panel_c_1.columns) + list(panel_c_2.columns)

tabluar_c = Tabular(("l" + "c" * nc) * nc, booktabs=True)
tabluar_c.add_row(
    (MultiColumn(ncol, align="l", data="Panel C: Time-Series Average of Industry Distribution"),)
)
tabluar_c.add_hline(1, ncol)
tabluar_c.add_row(header_c)
tabluar_c.add_hline(1, ncol)
for i in range(0, panel_c_1.shape[0]):
    tabluar_c.add_row(list(panel_c_1.iloc[i]) + list(panel_c_2.iloc[i]))

# Glue together
table = Table(position="htb!")

table.append(tabular)
table.append(tabluar_c)

tabular.generate_tex("../../04_results/summary_stats/underlying_summary_statistics")
table.generate_tex("../../08_figures/underlying_summary_statistics")

# %%
f = open("../../08_figures/underlying_summary_statistics" + ".tex")
lines = f.readlines()
f.close()

# insert resizebox
start_resizebox = "\\resizebox{\\textwidth}{!}{%\n"
end_resizebox = "}\\\\%\n"

idx_tabular = np.argwhere(["tabular" in line for line in lines]).T[0]
lines.insert(idx_tabular[0], start_resizebox)
lines.insert(idx_tabular[1] + 2, end_resizebox)
lines.insert(idx_tabular[2] + 2, start_resizebox)
lines.insert(idx_tabular[3] + 4, end_resizebox)

# kick second top rule
idx_tabular = np.argwhere(["\\toprule" in line for line in lines]).T[0]
del lines[idx_tabular[1]]

# kick first bottomrule
idx_tabular = np.argwhere(["\\bottomrule" in line for line in lines]).T[0]
bottomrule = lines[idx_tabular[0]]
lines[idx_tabular[0]] = bottomrule.replace("\\bottomrule", "")

with open("../../08_figures/underlying_summary_statistics" + ".tex", mode="w") as f:
    f.write("% generated by python\n")
f.close()

lines = lines[1:-1]
with open("../../08_figures/underlying_summary_statistics" + ".tex", mode="a") as f:
    for line in lines:
        f.write(line)
f.close()

# %%
