# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
""" Single model analysis.

Looks in-depth at the predictions of a single model.

Includes the following analyses:
    # # Sample importance
    # - Full model sample importance (i.e. feature importance-style)
    # - Within-model Diebold/Mariano test vs. subsamples
    # - Within-model forecast correlation

"""


# %%
import os
import pandas as pd
import numpy as np
from scoring import DieboldMariano, ForecastCorrelation, ClarkWest, XS_R2
from joblib import load

from pylatex import Tabular, MultiColumn, MultiRow, Math
from pylatex.utils import NoEscape

import matplotlib.pyplot as plt
import seaborn as sns

from analysis_setup import sns_palette, width, height, modelLoc, analysisLoc


class_groups = pd.read_parquet(os.path.join(analysisLoc, "class_groups.pq"))

sample_rename_dict = {
    "predicted": "score",
    "information-options": "O",
    "information-underlying": "S",
    "instrument-bucket;instrument-contract": "B+I",
    "N-En_information-options": "N-En: O",
    "N-En_information-underlying": "N-En: S",
    "N-En_instrument-bucket;instrument-contract": "N-En: B+I",
    "N-En_full": "N-En: full",
    "L-En_information-options": "L-En: O",
    "L-En_information-underlying": "L-En: S",
    "L-En_instrument-bucket;instrument-contract": "L-En: B+I",
    "L-En_full": "L-En: full",
}


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


# %% Read in new data, used from here on:
# ---- read in across-sample data for ensembles:
prediction_dict = load(os.path.join(modelLoc, "prediction_dict_samples.pkl"))
model_types = set([c.split("_")[0] for c in prediction_dict.keys()])


# %% Within-model R2 comparison --- interactions off vs. on (i.e. trained on full
#   sample vs. trained on subsamples):
to_plot = []
for model in prediction_dict:
    tmp = prediction_dict[model]["scores"].copy()
    tmp = tmp.to_frame()
    tmp.columns = ["score"]
    tmp["sample"] = model.split("_")[-1]
    tmp["model"] = model.split("_")[0]
    tmp = tmp.reset_index(drop=True)
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot)
to_plot["sample"] = to_plot["sample"].replace(sample_rename_dict.keys(), sample_rename_dict.values())
to_plot = to_plot.sort_values("sample")
to_plot = to_plot.set_index("sample")
to_plot = to_plot.loc[["full", "O", "B+I", "S"]].reset_index()
to_plot = to_plot[to_plot["model"] == "N-En"]

stars = []
for model in prediction_dict:
    if model.split("_")[0] != "N-En":
        continue
    predictions = prediction_dict[model]["predictions"]
    star = ClarkWest(predictions[["target", "predicted"]], 12, benchmark_type="zero", cw_adjust=False)
    star = star.to_frame()
    star.columns = ["stars"]
    star["sample"] = model.split("_")[-1]
    stars.append(star)

stars = pd.concat(stars)
stars["sample"] = stars["sample"].replace(sample_rename_dict.keys(), sample_rename_dict.values())
stars = stars.set_index("sample")
stars = stars.loc[["full", "O", "B+I", "S"]].reset_index(drop=True).squeeze()
stars = obtain_stars(stars)


fig, axes = plt.subplots(1, 1, figsize=(width, height), dpi=1000, sharex=True, sharey=True)
axes = (axes,)
for i, model in enumerate(["N-En"]):
    sns.barplot(
        x="sample",
        y="score",
        data=to_plot[to_plot["model"] == model],
        ax=axes[i],
        palette=sns_palette(4),
    )
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0, ha="center")
    axes[i].set_xlabel("")
    # axes[i].axhline(0, color="k", ls="--")
    axes[i].set_ylabel("")
    # axes[i].set_title(model)

coords = {
    (0.55, 0.022): ["Benefit of\nstock-based\nchar", 2.5, 0],
    (1.55, 0.0183): ["Benefit of\noption-based\nstock char", 1.8, 0],
    (3.55, 0.013): ["Benefit of option-based char", 12, 270],
}
for coord, text in coords.items():
    axes[-1].annotate(
        text[0],
        xy=coord,
        xytext=(coord[0] + 0.1, coord[1]),
        xycoords="data",
        fontsize=8,
        ha="left",
        va="center",
        rotation=text[2],
        arrowprops=dict(arrowstyle="-[, widthB=" + str(text[1]) + ", lengthB=.75", lw=1.0),
    )
axes[-1].set_xlim([-0.5, 3.8])

axes[-1].set_axisbelow(True)
for i, (_, txt) in enumerate(stars.iteritems()):
    axes[-1].annotate(txt, (i, 0.0247), ha="center")

fig.subplots_adjust(wspace=0.05)
fig.savefig("../08_figures/subsampling.pdf", bbox_inches="tight")


# %% XS R2 Sample importance
to_plot = []
for model in prediction_dict:
    tmp = prediction_dict[model]["predictions"][["target", "predicted"]]
    tmp = XS_R2(tmp)
    tmp = tmp.to_frame()
    tmp.columns = ["score"]
    tmp["sample"] = model.split("_")[-1]
    tmp["model"] = model.split("_")[0]
    tmp = tmp.reset_index(drop=True)
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot)
to_plot["sample"] = to_plot["sample"].replace(sample_rename_dict.keys(), sample_rename_dict.values())
to_plot = to_plot.sort_values("sample")
to_plot = to_plot.set_index("sample")
to_plot = to_plot.loc[["full", "O", "B+I", "S"]].reset_index()
to_plot = to_plot[to_plot["model"] == "N-En"]

stars = []
for model in prediction_dict:
    if model.split("_")[0] != "N-En":
        continue
    predictions = prediction_dict[model]["predictions"]
    star = ClarkWest(predictions[["target", "predicted"]], 12, benchmark_type="xs", cw_adjust=False)
    star = star.to_frame()
    star.columns = ["stars"]
    star["sample"] = model.split("_")[-1]
    stars.append(star)

stars = pd.concat(stars)
stars["sample"] = stars["sample"].replace(sample_rename_dict.keys(), sample_rename_dict.values())
stars = stars.set_index("sample")
stars = stars.loc[["full", "O", "B+I", "S"]].reset_index(drop=True).squeeze()
stars = obtain_stars(stars)

fig, axes = plt.subplots(1, 1, figsize=(width, height), dpi=1000, sharex=True, sharey=True)
axes = (axes,)
for i, model in enumerate(["N-En"]):
    sns.barplot(
        x="sample",
        y="score",
        data=to_plot[to_plot["model"] == model],
        ax=axes[i],
        palette=sns_palette(4),
    )
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0, ha="center")
    axes[i].set_xlabel("")
    # axes[i].axhline(0, color="k", ls="--")
    axes[i].set_ylabel("")
    # axes[i].set_title(model)

coords = {
    (0.55, 0.0285): ["Benefit of\nstock-based\nchar", 2.2, 0],
    (1.55, 0.0238): ["Benefit of\noption-based\nstock char", 2, 0],
    (3.55, 0.018): ["Benefit of option-based char", 11.5, 270],
}
for coord, text in coords.items():
    axes[-1].annotate(
        text[0],
        xy=coord,
        xytext=(coord[0] + 0.1, coord[1]),
        xycoords="data",
        fontsize=8,
        ha="left",
        va="center",
        rotation=text[2],
        arrowprops=dict(arrowstyle="-[, widthB=" + str(text[1]) + ", lengthB=.75", lw=1.0),
    )
axes[-1].set_xlim([-0.5, 3.8])

axes[-1].set_axisbelow(True)
for i, (_, txt) in enumerate(stars.iteritems()):
    axes[-1].annotate(txt, (i, 0.0315), ha="center")

fig.subplots_adjust(wspace=0.05)
fig.savefig("../08_figures/subsampling_xs_r2.pdf", bbox_inches="tight")


# %% Within-model DM test vs. subsamples
to_plot = []
for i, model in enumerate(prediction_dict):
    if i == 0:
        target = prediction_dict[model]["predictions"]["target"].copy()
    tmp = prediction_dict[model]["predictions"]["predicted"]
    tmp.name = model
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot = pd.concat((to_plot, target), axis=1)
to_plot.columns = [sample_rename_dict[c] if c in sample_rename_dict.keys() else c for c in to_plot.columns]
to_plot = to_plot.reindex(sorted(to_plot.columns), axis=1)

diebold_test = DieboldMariano(to_plot, 12)
diebold_test.insert(3, "", np.nan)


def latex_table(data, save_name: str, num_format="%.4f"):
    """Generates latex table from pd.DataFrame

    Args:
        data (pd.DataFrame): input data
        save_name (str): save name of table
        skip_index (bool, optional): Whether to skip index when creating table. Defaults to False.
        num_format (str, optional): Format for numbers.
    """

    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    table = Tabular("".join(["l"] + ["c"] * (data.shape[1])), booktabs=True)
    table.add_row(
        (
            "",
            MultiColumn(
                (len(data.columns)) // 2 - 1,
                align="c",
                data=[c.split(": ")[0] for c in data.columns.tolist()][0],
            ),
            "",
            MultiColumn(
                (len(data.columns)) // 2,
                align="c",
                data=[c.split(": ")[0] for c in data.columns.tolist()][-1],
            ),
        )
    )

    table.add_hline(start=2, end=4)
    table.add_hline(start=6, end=9)
    table.add_row([""] + [c.split(": ")[-1] for c in data.columns.tolist()])
    table.add_hline(start=2, end=len(data.columns) + 1)

    added_hline = False
    for idx, row in data.iterrows():
        if (idx.split(": ")[0] == "N-En") & (added_hline is False):
            table.add_hline()
            added_hline = True
        to_add = []
        for num in row:
            if np.isnan(num):
                to_add.append("")
            else:
                if np.abs(num) > 2.56:
                    num = num_format % num
                    to_add.append(math(r"\textcolor{Cyan}{\textbf{%s}}" % num))
                elif np.abs(num) > 1.96:
                    num = num_format % num
                    to_add.append(math(r"\textcolor{Blue}{\textbf{%s}}" % num))
                else:
                    to_add.append(math(num_format % num))
        table.add_row([idx] + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


latex_table(diebold_test, "dm_subsamples", num_format="%.2f")


# %% Within-model forecast correlation
to_plot = []
for model in prediction_dict:
    tmp = prediction_dict[model]["predictions"]["predicted"].copy()
    tmp.name = model
    to_plot.append(tmp.copy())
to_plot = pd.concat(to_plot, axis=1)
to_plot.columns = [sample_rename_dict[c] for c in to_plot.columns]
to_plot = to_plot.reindex(sorted(to_plot.columns), axis=1)

forecast_correlation = ForecastCorrelation(to_plot)
forecast_correlation.insert(3, "", np.nan)


def latex_table(data, save_name: str, num_format="%.4f"):
    """Generates latex table from pd.DataFrame

    Args:
        data (pd.DataFrame): input data
        save_name (str): save name of table
        skip_index (bool, optional): Whether to skip index when creating table. Defaults to False.
        num_format (str, optional): Format for numbers.
    """

    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    table = Tabular("".join(["l"] + ["c"] * (data.shape[1])), booktabs=True)
    table.add_row(
        (
            "",
            MultiColumn(
                (len(data.columns)) // 2 - 1,
                align="c",
                data=[c.split(": ")[0] for c in data.columns.tolist()][0],
            ),
            "",
            MultiColumn(
                (len(data.columns)) // 2,
                align="c",
                data=[c.split(": ")[0] for c in data.columns.tolist()][-1],
            ),
        )
    )
    table.add_hline(start=2, end=4)
    table.add_hline(start=6, end=9)
    table.add_row([""] + [c.split(": ")[-1] for c in data.columns.tolist()])
    table.add_hline(start=2, end=len(data.columns) + 1)

    added_hline = False
    for idx, row in data.iterrows():
        if (idx.split(": ")[0] == "N-En") & (added_hline is False):
            table.add_hline()
            added_hline = True
        to_add = []
        for num in row:
            if np.isnan(num):
                to_add.append("")
            else:
                if num > 0.9:
                    num = num_format % num
                    to_add.append(math(r"\textcolor{Cyan}{\textbf{%s}}" % num))
                elif num > 0.7:
                    num = num_format % num
                    to_add.append(math(r"\textcolor{Blue}{\textbf{%s}}" % num))
                else:
                    to_add.append(math(num_format % num))
        table.add_row([idx] + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


latex_table(forecast_correlation, "forecast_correlation_subsamples", num_format="%.2f")


# %% Within-model sample performance -- per option bucket
to_plot = []
sig_stars = []
for model in prediction_dict:
    tmp = prediction_dict[model]["predictions"]
    tmp = tmp.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])
    tmp = tmp.set_index("date")
    tmp = tmp[["target", "predicted", "bucket"]]
    cw_test = tmp.groupby("bucket").apply(lambda x: ClarkWest(x, 12, benchmark_type="zero", cw_adjust=False))
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
    stars = pd.Series(data=stars, index=cw_test.index, name=sample_rename_dict[model])
    stars = stars[~stars.index.str.contains("nan")]

    tmp = prediction_dict[model]["errors"]["bucket"].copy()
    tmp = tmp.reset_index(level="bucket")
    tmp = tmp[~tmp.bucket.str.contains("nan")]
    scores = (
        tmp.groupby("bucket")[[c for c in ["predicted", "target"]]]
        .apply(lambda x: 1 - x.sum() / x["target"].sum())
        .drop(columns=["target"])
    )
    scores = scores.reset_index()
    scores.columns = [c.strip("m_") for c in scores.columns]

    # rearrange to put in table:
    ttm = []
    moneyness = []
    types = []
    for i, bucket in enumerate(scores.bucket.str.split("_").values):
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
    tmp = pd.DataFrame()
    tmp["TTM"] = ttm
    tmp["Mon."] = moneyness
    tmp["Type"] = types
    tmp = pd.concat((tmp, scores), axis=1)
    tmp["stars"] = stars.values
    tmp = tmp.drop(columns=["bucket"])
    tmp = tmp.sort_values("TTM", ascending=False)
    tmp = tmp.set_index(["TTM", "Mon.", "Type"])
    name = sample_rename_dict[model].split(": ")
    stars = tmp["stars"]
    tmp = tmp.drop(columns=["stars"])
    tmp.columns = pd.MultiIndex.from_arrays(([name[0]], [name[1]]))
    to_plot.append(tmp)
    sig_stars.append(stars)

to_plot = pd.concat(to_plot, axis=1)
sig_stars = pd.concat(sig_stars, axis=1)
sig_stars.columns = to_plot.columns

to_plot = to_plot.reindex(sorted(to_plot.columns), axis=1)
sig_stars = sig_stars.reindex(sorted(sig_stars.columns), axis=1)

to_plot.insert(4, "", np.nan)
sig_stars.insert(4, "", np.nan)


def latex_table(data, sig_stars, save_name: str, num_format="%.4f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    table = Tabular("".join(["l"] * len(data.index[0]) + ["r"] * (data.shape[1])), booktabs=True)
    table.add_row(
        [""] * len(data.index[0])
        + [
            MultiColumn(
                (len(data.columns) - 1) // 2,
                align="c",
                data=data.columns.get_level_values(0)[0],
            ),
            "",
            MultiColumn(
                (len(data.columns) - 1) // 2,
                align="c",
                data=data.columns.get_level_values(0)[-3],
            ),
        ]
    )
    table.add_hline(start=4, end=7)
    table.add_hline(start=9, end=12)
    to_add = data.index.names + data.columns.get_level_values(1).tolist()
    to_add = ["" if c == "Type" else c for c in to_add]
    table.add_row([MultiColumn(1, align="c", data=c) for c in to_add])

    for i, (idx, row) in enumerate(data.iterrows()):
        to_add = []
        for j, (col, num) in enumerate(row.iteritems()):
            if np.isnan(num):
                to_add.append("")
            else:
                if num == row.loc[col[0]].max():
                    # num = num_format % num
                    # to_add.append(math(r"\textcolor{Blue}{%s}" % num))
                    num = num_format % num
                    # if sig_stars.iloc[i, j] in ["**", "***"]:
                    #     to_add.append(math(r"\textcolor{Cyan}{\underline{%s}}" % num))
                    # else:
                    #     to_add.append(math(r"\underline{%s}" % num))
                    # to_add.append(r"\underline{%s}" % (num + sig_stars.iloc[i, j]))
                    to_add.append(r"%s" % (num + sig_stars.iloc[i, j]))
                else:
                    num = num_format % num
                    # if sig_stars.iloc[i, j] in ["**", "***"]:
                    #     to_add.append(math(r"\textcolor{Cyan}{%s}" % num))
                    # else:
                    #     to_add.append(math(num))
                    to_add.append(r"%s" % (num + sig_stars.iloc[i, j]))
        if i % 5 == 0:
            table.add_hline()
            table.add_row([MultiRow(5, data=math(idx[0]))] + list(idx[1:]) + to_add)
        else:
            table.add_row([""] + list(idx[1:]) + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


latex_table(to_plot, sig_stars, "bucket_subsample_performance", num_format="%.3f")

file = "bucket_subsample_performance"
latex_table(to_plot, sig_stars, "bucket_subsample_performance", num_format="%.3f")

with open("../08_figures/" + file + ".tex") as f:
    lines = f.readlines()

N = to_plot.shape[1]
lines[0] = lines[0].replace(
    "r" * N,
    r"S[table-format = 2.3,table-space-text-post={*}]" * N + "!{\quad}",
)

with open("../08_figures/" + file + ".tex", "w") as f:
    f.writelines(lines)
