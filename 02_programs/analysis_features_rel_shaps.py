# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

import os
import pandas as pd
import numpy as np
from scoring import ClarkWest, DieboldMariano, DieboldMariano_XS, ForecastCorrelation, XS_R2

from pylatex import Tabular, MultiColumn, MultiRow, Math
from pylatex.utils import NoEscape

import matplotlib.pyplot as plt

# import matplotlib
import seaborn as sns

from analysis_setup import (
    prediction_dict,
    sns_palette,
    width,
    height,
    group_dict,
    analysisLoc,
    # n_top_features,
    # n_bot_features,
)


from shap._explanation import Explanation
import shap
import matplotlib.pyplot as plt
import matplotlib

def append_group_name(feature, feature_class):
    gp = group_abbreviations(feature_class[feature_class.Feature == feature].copy().Group.iloc[0])
    return "{} ({})".format(feature, gp)

def group_abbreviations(grp):
    if grp == "Accruals":
        idx = 3
    elif grp == "Profitability":
        idx = 4
    elif grp == "Quality":
        idx = 1
    elif grp == "Investment":
        idx = 3
    elif grp == "Illiquidity":
        idx = 3
    elif grp == "Informed Trading":
        idx = 4
    elif grp == "Value":
        idx = 3
    elif grp == "Contract":
        idx = 1
    elif grp == "Past Prices":
        idx = 4
    elif grp == "Frictions":
        idx == 4
    elif grp == "Industry":
        idx == 3
    elif grp == "Risk":
        idx = 4
    return grp[:idx]

def transformer(a, b, x):
    beta = 99 / (b - a)
    alpha = -beta * a
    return alpha + beta * x


def scaler(row):
    p = row["percentile"]
    a = row["scaling"][0]
    b = row["scaling"][1]
    new_p = transformer(a, b, p)
    return new_p


def do_scaling(tmp, cols):
    scaling_factor = (
        tmp.dropna()
        .groupby(cols, as_index=False)
        .filter(lambda x: x.shape[0] > 1)
        .groupby(cols)
        .apply(lambda x: (x.percentile.min(), x.percentile.max()))
    )
    scaling_factor.name = "scaling"
    tmp = tmp.merge(scaling_factor, right_index=True, left_on=cols)
    tmp["percentile"] = tmp.apply(lambda row: scaler(row), axis=1)
    tmp.percentile = tmp.percentile.apply(lambda x: min(x, 99))
    tmp.drop(columns="scaling", inplace=True)
    return tmp


vol_jump_risks = ["tlm30", "ivvol", "volunc", "skewiv", "rns30", "rnk30", "gamma", "vega", "ivrv"]
model = "N-En"

feature_class = pd.read_parquet("../03_data/features_overview.pq")

# %%
# Beeswarm plot for 10 most important characteristics over entire model
num_feat = 10
scale_flag = True

# identify 10 most important characteristics

full = prediction_dict[model]["SHAPs"]["full"].copy()

N = full.N
full = full[[c for c in full.columns if not c.startswith("m_")]]
full = full[
    [
        c
        for c in full.columns
        if c
        not in [
            "m_information-options",
            "m_information-underlying",
            "m_instrument-bucket;instrument-contract",
            "m_degree",
            "N",
        ]
    ]
]
full = full.multiply(N, axis=0)  # scale to sum of SHAPs
means = full.sum()
means = means.divide(means.sum())
means = means.sort_values(ascending=False).head(num_feat)

# extract relative SHAPs and plot beeswarm

tmp = prediction_dict[model]["conditional_SHAPs"]["single"].copy()

tmp = tmp[[c for c in means.index.values]]
tmp = tmp.stack(dropna=False).reset_index()
tmp.rename({0: "shap"}, axis=1, inplace=True)

if scale_flag:
    tmp = do_scaling(tmp, ["level_2", "date"])

tmp.set_index("date", inplace=True)

s_values = pd.DataFrame()
d_values = pd.DataFrame()
for c in means.index.values:
    t = tmp.loc[tmp.level_2 == c, :]
    c = append_group_name(c, feature_class)
    s_values[c] = t["shap"].values
    d_values[c] = t.percentile.apply(lambda x: min(x, 99)).values

# impact
shaps = Explanation(values=s_values.values, data=d_values.values, feature_names=s_values.columns.values)

fig = shap.plots.beeswarm(
    shaps,
    order = None,
    color=sns.color_palette("flare", as_cmap=True),
    plot_size=(width, height),
)
fig.tight_layout()
fig.savefig("../08_figures/shaps_single_direction.pdf")


# %%
# do plot over time
fig, ax = plt.subplots(5, 2, figsize=(width, 18 / 2.54), sharex=True, dpi=800)

scatter_plot_dot = 5

n = 0
for c in means.index.values:
    j = np.floor_divide(n, 5)
    i = np.remainder(n, 5)
    t = tmp.loc[tmp.level_2 == c, :]
    t.index = t.index.to_timestamp()
    t.reset_index(inplace=True)
    t.percentile = t.percentile.round().astype(int)
    t.dropna(inplace=True)
    t = t[(t.percentile % 20 == 0) | (t.percentile == 99)]
    # cutter = t.groupby(["date"]).apply(lambda x: x.percentile.values[1::2])
    # t = t.loc[t.percentile.mod(2) == 0,:]
    sns.scatterplot(
        x="date",
        y="shap",
        data=t,
        hue=t.percentile.values,
        legend=None,
        palette=sns.color_palette("flare", as_cmap=True),
        ax=ax[i, j],
        s=scatter_plot_dot,
    )
    ax[i, j].set_ylabel("")
    ax[i, j].set_xlabel("")
    ax[i, j].set_title(c, fontsize=8)
    # fig.autofmt_xdate()
    ax[i, j].xaxis.set_major_locator(matplotlib.dates.YearLocator(4))
    ax[i, j].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    n += 1

fig.tight_layout()
fig.savefig("../08_figures/shaps_single_time.pdf")

# %%
# do plot for functional form

tmp = prediction_dict[model]["conditional_SHAPs"]["single"].copy()

tmp = tmp.stack(dropna=False).reset_index()
tmp.rename({0: "shap"}, axis=1, inplace=True)

if scale_flag:
    tmp = do_scaling(tmp, ["level_2", "date"])
    tmp.percentile = tmp.percentile.round()

tmp = tmp.drop(columns="date").groupby(["percentile", "level_2"]).mean().reset_index()

fig, ax = plt.subplots(5, 2, figsize=(width, 18 / 2.54), sharex=True, dpi=800)
scatter_plot_dot = 8
n = 0
for c in means.index.values:
    j = np.floor_divide(n, 5)
    i = np.remainder(n, 5)
    t = tmp.loc[tmp.level_2 == c, :]
    t = t[(t.percentile % 2 == 0) | (t.percentile == 99)]
    sns.scatterplot(
        x="percentile",
        y="shap",
        data=t,
        # hue = t.percentile.values,
        legend=None,
        # palette = sns.color_palette("flare", as_cmap=True),
        ax=ax[i, j],
        color="black",
        s=12,
    )

    if i == 4:
        ax[i, j].set_xlabel("Percentile")
    else:
        ax[i, j].set_xlabel("")

    ax[i, j].set_ylabel("")
    ax[i, j].set_title(c, fontsize=8)
    # fig.autofmt_xdate()
    n += 1

fig.tight_layout()
fig.savefig("../08_figures/shaps_single_functionalform.pdf")

# %% 
# do plot for functional form for L-En and N-En

form= []
for model in ["L-En","N-En"]:
    tmp = prediction_dict[model]["conditional_SHAPs"]["single"].copy()
    
    tmp = tmp.stack(dropna=False).reset_index()
    tmp.rename({0: "shap"}, axis=1, inplace=True)
    
    if scale_flag:
        tmp = do_scaling(tmp, ["level_2", "date"])
        tmp.percentile = tmp.percentile.round()
    
    tmp = tmp.drop(columns="date").groupby(["percentile", "level_2"]).mean().reset_index()
    tmp["model"] = model
    form.append(tmp)
form = pd.concat(form)

colors = sns.color_palette("hls", n_colors=2)
legend_fontsize = "6"
fig, ax = plt.subplots(5, 2, figsize=(width, 18 / 2.54), sharex=True, dpi=800)
scatter_plot_dot = 8
n = 0
for c in means.index.values:
    j = np.floor_divide(n, 5)
    i = np.remainder(n, 5)

    t = form.loc[(form.level_2 == c) , :]
    t = t[(t.percentile % 2 == 0) | (t.percentile == 99)]
    
    if i==4:
        legend_flag = True
    else:
        legend_flag = False
        
    sns.scatterplot(
        x="percentile",
        y="shap",
        data=t,
        hue = "model",
        legend=legend_flag,
        # palette = sns.color_palette("flare", as_cmap=True),
        ax=ax[i, j],
        palette=colors,
        s=12,
    )
    
    if i == 4:
        ax[i, j].set_xlabel("Percentile")
        ax[i, j].legend().set_title("")
        ax[i, j].legend(frameon=False)
        plt.setp(ax[i, j].get_legend().get_texts(), fontsize=legend_fontsize)
    else:
        ax[i, j].set_xlabel("")

    ax[i, j].set_ylabel("")
    ax[i, j].set_title(c, fontsize=8)
    # fig.autofmt_xdate()
    n += 1

fig.tight_layout()
fig.savefig("../08_figures/shaps_single_functionalform_both.pdf")

# %%
# Jump and stochastic vol risk and their impact on returns
def math(x):
    return Math(data=[NoEscape(x)], inline=True)


jump_risks = ["tlm30", "skewiv", "rns30", "rnk30", "gamma"]
vol_risks = ["ivvol", "volunc", "vega", "ivrv"]

# do table with comparison of their importance compared to full model

full = prediction_dict[model]["SHAPs"]["full"].copy()
N = full.N
full = full[[c for c in full.columns if not c.startswith("m_")]]
full = full[
    [
        c
        for c in full.columns
        if c
        not in [
            "m_information-options",
            "m_information-underlying",
            "m_instrument-bucket;instrument-contract",
            "N",
        ]
    ]
]
full = full.multiply(N, axis=0)  # scale to sum of SHAPs
full = full.sum()
full = full.divide(full.sum())
full = full.sort_values(ascending=False).rank(ascending=False)
full.name = "full"
full = full.loc[full.index.isin(vol_risks + jump_risks)]

to_show = []
tmp = prediction_dict[model]["SHAPs"]["bucket"].copy()
tmp.reset_index("bucket", inplace=True)
tmp = tmp[~tmp.bucket.str.contains("nan")]
for idx, ttemp in tmp.groupby("bucket"):
    ttemp.drop(columns="bucket", inplace=True)
    N = ttemp.N
    ttemp = ttemp[[c for c in ttemp.columns if not c.startswith("m_")]]
    ttemp = ttemp[
        [
            c
            for c in ttemp.columns
            if c
            not in [
                "m_information-options",
                "m_information-underlying",
                "m_instrument-bucket;instrument-contract",
                "N",
            ]
        ]
    ]
    ttemp = ttemp.multiply(N, axis=0)  # scale to sum of SHAPs
    means = ttemp.sum()
    means = means.divide(means.sum())
    means = means.sort_values(ascending=False).rank(ascending=False)
    means.name = idx

    t = means.loc[means.index.isin(jump_risks + vol_risks)]
    to_show.append(pd.DataFrame(t).T)

to_show = pd.concat(to_show)

to_show = to_show.subtract(full)


ttm = []
moneyness = []
types = []
for _, bucket in enumerate(to_show.index.str.split("_").values):
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
to_show["TTM"] = ttm
to_show["Mon."] = moneyness
to_show["Type"] = types

to_show = to_show.sort_values(["TTM", "Mon.", "Type"], ascending=[False, True, True])
to_show = to_show.set_index(["TTM", "Mon.", "Type"])

data = to_show[vol_risks]
data[""] = ""
data = pd.concat((data, to_show[jump_risks]), axis=1)
all = full[vol_risks]
all.loc[""] = ""
all = pd.concat((all, full[jump_risks]))

num_format = "%.0f"
save_name = "option_risk_premia"
group_names = ["Volatility risk", "Jump risk"]

N = len(data.index[0]) + data.shape[1]
table = Tabular("".join(["l"] * len(data.index[0]) + ["c"] * (data.shape[1])), booktabs=True)
table.add_row(
    [""] * len(data.index[0])
    + [MultiColumn(len(vol_risks), align="c", data=group_names[0])]
    + [""]
    + [MultiColumn(len(jump_risks), align="c", data=group_names[1])]
)
table.add_row([""] * len(data.index[0]) + list(data.columns.values))
table.add_hline()

to_add = []
for col, num in all.iteritems():
    if num == "":
        to_add.append("")
    else:
        to_add.append(math(num_format % num))
table.add_row([MultiColumn(3, align="r", data="all Sample")] + to_add)

table.add_empty_row()
table.add_row((MultiColumn(N, align="c", data="Buckets"),))

for i, (idx, row) in enumerate(data.iterrows()):
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
        table.add_row([MultiRow(5, data=math(idx[0]))] + list(idx[1:]) + to_add)
    else:
        table.add_row([""] + list(idx[1:]) + to_add)

table.generate_tex("../08_figures/%s" % save_name)


# %%
# analyse of vol and jump risk for different buckets
def bucket_renamer(inp):
    return inp.replace("_", " ").replace("call", "C").replace("put", "P")

scale_flag = True

tmp = prediction_dict[model]["conditional_SHAPs"]["bucket"].copy()
tmp = tmp.stack().reset_index()
tmp.rename({"level_3": "feature", 0: "shap"}, axis=1, inplace=True)
tmp = tmp[tmp.feature.isin(vol_jump_risks)]
tmp = tmp[~tmp.bucket.str.contains("nan")]

if scale_flag:
    tmp = do_scaling(tmp, ["date", "bucket", "feature"])
    tmp.percentile = tmp.percentile.round()
tmp = tmp.drop(columns="date").groupby(["percentile", "bucket", "feature"]).mean().reset_index()
tmp_short = tmp.loc[tmp.bucket.apply(lambda x: "short" in x), :]
tmp_short.bucket = tmp_short.bucket.apply(lambda x: bucket_renamer(x))
tmp_long = tmp.loc[tmp.bucket.apply(lambda x: "long" in x), :]
tmp_long.bucket = tmp_long.bucket.apply(lambda x: bucket_renamer(x))

# %%
def plotter(f1, f2, name):
    fig, ax = plt.subplots(2, 2, figsize=(width, 10 / 2.54), sharex=True, dpi=800)
    colors = sns.color_palette("hls", n_colors=5)

    legend_fontsize = "6"
    scatter_plot_dot = 15

    data = tmp_short.loc[tmp_short.feature == f1, :]
    data = data[(data.percentile % 5 == 0) | (data.percentile == 99)]
    sns.scatterplot(
        x="percentile",
        y="shap",
        hue="bucket",
        data=data,
        palette=colors,
        ax=ax[0, 0],
        legend=None,
        s=scatter_plot_dot,
        edgecolor="none",
    )
    sns.lineplot(
        x="percentile",
        y="shap",
        hue="bucket",
        data=data,
        palette=colors,
        ax=ax[0, 0],
        legend=None,
        linestyle="--",
        linewidth=0.5,
    )

    ax[0, 0].set_ylabel("")
    ax[0, 0].set_xlabel("")
    ax[0, 0].set_title(f1, fontsize=8)

    data = tmp_short.loc[tmp_short.feature == f2, :]
    data = data[(data.percentile % 5 == 0) | (data.percentile == 99)]
    sns.scatterplot(
        x="percentile",
        y="shap",
        hue="bucket",
        data=data,
        ax=ax[1, 0],
        legend=True,
        palette=colors,
        s=scatter_plot_dot,
        edgecolor="none",
    )
    sns.lineplot(
        x="percentile",
        y="shap",
        hue="bucket",
        data=data,
        palette=colors,
        ax=ax[1, 0],
        legend=None,
        linestyle="--",
        linewidth=0.5,
    )

    ax[1, 0].legend().set_title("")
    ax[1, 0].legend(frameon=False)
    plt.setp(ax[1, 0].get_legend().get_texts(), fontsize=legend_fontsize)

    ax[1, 0].set_ylabel("")
    ax[1, 0].set_xlabel("Percentile")
    ax[1, 0].set_title(f2, fontsize=8)

    data = tmp_long.loc[tmp_long.feature == f1, :]
    data = data[(data.percentile % 5 == 0) | (data.percentile == 99)]
    sns.scatterplot(
        x="percentile",
        y="shap",
        hue="bucket",
        data=data,
        ax=ax[0, 1],
        palette=colors,
        legend=None,
        s=scatter_plot_dot,
        edgecolor="none",
    )
    sns.lineplot(
        x="percentile",
        y="shap",
        hue="bucket",
        data=data,
        palette=colors,
        ax=ax[0, 1],
        legend=None,
        linestyle="--",
        linewidth=0.5,
    )

    ax[0, 1].set_ylabel("")
    ax[0, 1].set_xlabel("")
    ax[0, 1].set_title(f1, fontsize=8)

    data = tmp_long.loc[tmp_long.feature == f2, :]
    data = data[(data.percentile % 5 == 0) | (data.percentile == 99)]
    sns.scatterplot(
        x="percentile",
        y="shap",
        hue="bucket",
        data=data,
        ax=ax[1, 1],
        palette=colors,
        legend=True,
        s=scatter_plot_dot,
        edgecolor="none",
    )
    sns.lineplot(
        x="percentile",
        y="shap",
        hue="bucket",
        data=data,
        palette=colors,
        ax=ax[1, 1],
        legend=None,
        linestyle="--",
        linewidth=0.5,
    )

    ax[1, 1].legend().set_title("")
    ax[1, 1].legend(frameon=False)
    plt.setp(ax[1, 1].get_legend().get_texts(), fontsize=legend_fontsize)

    ax[1, 1].set_ylabel("")
    ax[1, 1].set_xlabel("Percentile")
    ax[1, 1].set_title(f2, fontsize=8)

    fig.tight_layout()
    fig.savefig(name)


name = "../08_figures/shaps_risk_premia_breakdown.pdf"
plotter("ivrv", "gamma", name)
plotter("vega", "tlm30", name.replace(".pdf", "2.pdf"))