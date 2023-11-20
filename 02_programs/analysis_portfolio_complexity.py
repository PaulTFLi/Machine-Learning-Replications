# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
""" Portfolio analysis by means of single SHAPs.

Includes the following analyses:
    # - compute complexity for each decile portfolio
    #   - number of important features
    #   - nonlinearities
    #   - interactions

"""

import pandas as pd
import numpy as np

from pylatex import Tabular, Math
from pylatex.utils import NoEscape

from analysis_setup import prediction_dict


from joblib import delayed, Parallel
import glob


def iqr(x):
    if len(x) >= 4:
        return np.quantile(x, 0.75) - np.quantile(x, 0.25)
    else:
        return np.nan


def local_dispersion(gp, chars):
    gp = gp.reset_index()
    port = gp.port.iloc[0]
    date = gp.date.iloc[0]
    gp = gp.drop(columns="port")
    gp = gp.set_index(["date", "optionid"])
    importance = gp.abs().mean() / gp.abs().mean().sum()
    importance = importance[importance > n_important_cutoff]
    chars_names = importance.index.values
    gp = gp.merge(chars, left_index=True, right_index=True, suffixes=("_shap", ""))
    data = []
    for char in chars_names:
        tmp = gp[[char, f"{char}_shap"]]
        tmp["bin"] = pd.cut(tmp[char], np.linspace(-1, 1, 101), duplicates="drop").apply(lambda x: x.mid)
        tmp["perc"] = pd.qcut(tmp[char], 100, duplicates="drop").apply(lambda x: x.mid)
        tmp_std = tmp.groupby("bin").apply(lambda x: np.nanstd(x[f"{char}_shap"]))
        tmp_iqr = tmp.groupby("bin").apply(lambda x: iqr(x[f"{char}_shap"]))
        N = tmp.groupby("bin").apply(lambda x: x.shape[0])
        tmp_std = np.nansum((tmp_std * N) / np.nansum(N))
        tmp_iqr = np.nansum((tmp_iqr * N) / np.nansum(N))

        tmp_std_perc = tmp.groupby("perc").apply(lambda x: np.nanstd(x[f"{char}_shap"]))
        tmp_iqr_perc = tmp.groupby("perc").apply(lambda x: iqr(x[f"{char}_shap"]))
        N = tmp.groupby("perc").apply(lambda x: x.shape[0])
        if tmp_std_perc.empty:
            tmp_std_perc = 0
        else:
            tmp_std_perc = np.nansum((tmp_std_perc * N) / np.nansum(N))
        if tmp_iqr_perc.empty:
            tmp_iqr_perc = 0
        else:
            tmp_iqr_perc = np.nansum((tmp_iqr_perc * N) / np.nansum(N))

        tmp = pd.DataFrame(
            {
                "plain_std": tmp_std,
                "plain_iqr": tmp_iqr,
                "perc_std": tmp_std_perc,
                "perc_iqr": tmp_iqr_perc,
                "char": char,
            },
            index=[0],
        )
        data.append(tmp)
    data = pd.concat(data)
    data = data.reset_index(drop=True)
    importance.name = "importance"
    data = pd.merge(data, importance, right_index=True, left_on=["char"])
    plain_std = np.sum(data.importance * data.plain_std / data.importance.sum())
    plain_iqr = np.sum(data.importance * data.plain_iqr / data.importance.sum())
    perc_std = np.sum(data.importance * data.perc_std / data.importance.sum())
    perc_iqr = np.sum(data.importance * data.perc_iqr / data.importance.sum())

    return pd.DataFrame(
        {
            "plain_std": plain_std,
            "plain_iqr": plain_iqr,
            "perc_std": perc_std,
            "perc_iqr": perc_iqr,
            "date": date,
            "port": port,
        },
        index=[0],
    )


from skfda import FDataGrid
import skfda.preprocessing.smoothing.kernel_smoothers as ks
import statsmodels.api as sm


def smooth_and_fit(gp, chars):
    gp = gp.reset_index()
    port = gp.port.iloc[0]
    date = gp.date.iloc[0]
    gp = gp.drop(columns="port")
    gp = gp.set_index(["date", "optionid"])
    importance = gp.abs().mean() / gp.abs().mean().sum()
    importance = importance[importance > n_important_cutoff]
    chars_names = importance.index.values
    gp = gp[chars_names]
    gp = gp.merge(chars[chars_names], left_index=True, right_index=True, suffixes=("_shap", ""))
    data = []
    for char in chars_names:
        tmp = gp[[char, f"{char}_shap"]]
        tmp = tmp.groupby([char]).mean()
        fd = FDataGrid(sample_points=[tmp.index.values], data_matrix=[tmp[f"{char}_shap"].values])
        smoother = ks.NadarayaWatsonSmoother(smoothing_parameter=0.05)
        smoothed = smoother.fit_transform(fd)

        y = smoothed.data_matrix[0]
        x = smoothed.grid_points[0]
        x = sm.add_constant(x, prepend=False)

        res = sm.OLS(y, x).fit()
        std_res = np.std(res.resid)

        tmp = pd.DataFrame({"std_res": std_res, "char": char}, index=[0])
        data.append(tmp)
    data = pd.concat(data)
    data = data.reset_index(drop=True)
    importance.name = "importance"
    data = pd.merge(data, importance, right_index=True, left_on=["char"])
    std_res = np.sum(data.importance * data.std_res / data.importance.sum())

    return pd.DataFrame({"std_res": std_res, "date": date, "port": port}, index=[0])


files_shap = glob.glob("../04_results/shaps/*.pq")
files_sample = glob.glob("../04_results/option_sample/characteristics_*.pq")

n_important_cutoff = 0.01

model = "N-En"
n_important = []
interaction_effects = []
non_linearities = []
pred = prediction_dict[model]["predictions"].copy()
n_obs = pred.groupby(["date", "port"]).size()
pred = pred[["optionid", "port"]].reset_index().set_index(["date", "optionid"])
files_model = [x for x in files_shap if model.upper().replace("-", "_") in x]

for file in files_model:
    print(f"Working on {file}")
    # file = files_model[-1]
    shaps = pd.read_parquet(file)
    shaps = shaps.reset_index()
    year = shaps.date.iloc[0].year
    shaps["date"] = shaps["date"].dt.to_period("M")
    shaps = shaps.set_index(["date", "optionid"])
    shaps = shaps.merge(pred, left_index=True, right_index=True)
    shaps = shaps.reset_index().set_index(["date", "optionid", "port"])
    feature_importance = shaps.groupby(["date", "port"]).apply(lambda x: x.abs().mean() / x.abs().mean().sum())

    # number of important features (rel. importance above 1%)
    n_important_tmp = (feature_importance > n_important_cutoff).sum(axis=1)
    n_important.append(n_important_tmp)

    # get raw characteristics
    chars = pd.concat([pd.read_parquet(f) for f in files_sample if str(year) in f])
    chars = chars.iloc[:, :274]
    chars.index = chars.index.to_period("M")
    chars = chars.reset_index().set_index(["date", "optionid"])

    interaction_effects_tmp = pd.concat(
        Parallel(n_jobs=6, backend="threading", verbose=True)(
            delayed(local_dispersion)(gp, chars.copy()) for _, gp in shaps.groupby(["date", "port"])
        )
    )
    interaction_effects.append(interaction_effects_tmp)
    non_linearities_tmp = pd.concat(
        Parallel(n_jobs=6, backend="threading", verbose=True)(
            delayed(smooth_and_fit)(gp, chars.copy()) for _, gp in shaps.groupby(["date", "port"])
        )
    )
    non_linearities.append(non_linearities_tmp)

n_important = pd.concat(n_important)
n_important.name = "n_imp"
interaction_effects = pd.concat(interaction_effects)
non_linearities = pd.concat(non_linearities)

n_weigthing = n_obs.groupby("port").apply(lambda x: x / x.sum())
n_weigthing.name = "weighting"

n_important = pd.merge(n_important, n_weigthing, left_index=True, right_index=True)
n_important = n_important.groupby("port").apply(lambda x: (x.n_imp * x.weighting).sum())
n_important.name = "Nbr Imp. Features"

non_linearities = non_linearities.set_index(["date", "port"])
non_linearities = pd.merge(non_linearities, n_weigthing, left_index=True, right_index=True)
non_linearities = non_linearities.groupby("port").apply(lambda x: (x.std_res * x.weighting).sum())
non_linearities.name = "Nonlinearities"

interaction_effects = interaction_effects.set_index(["date", "port"])
interaction_effects = pd.merge(interaction_effects, n_weigthing, left_index=True, right_index=True)
interaction_effects_std = interaction_effects.groupby("port").apply(lambda x: (x["plain_std"] * x.weighting).sum())
interaction_effects_std.name = "Interactions (Std)"
interaction_effects_iqr = interaction_effects.groupby("port").apply(lambda x: (x["plain_iqr"] * x.weighting).sum())
interaction_effects_iqr.name = "Interactions (IQR)"

output = pd.merge(n_important, non_linearities, left_index=True, right_index=True)
output = pd.merge(output, interaction_effects_std, left_index=True, right_index=True)
output = pd.merge(output, interaction_effects_iqr, left_index=True, right_index=True)

output.index = "Lo 2 3 4 5 6 7 8 9 Hi".split()
output.loc[:, ["Nonlinearities", "Interactions (Std)", "Interactions (IQR)"]] *= 100


def latex_table(full_sample, save_name: str, num_format="%.4f"):
    def math(x):
        return Math(data=[NoEscape(x)], inline=True)

    table = Tabular("".join(["l"] + ["c"] * full_sample.shape[1]), booktabs=True)
    table.add_row([""] + list(full_sample.columns))

    table.add_hline()

    for i, (idx, row) in enumerate(full_sample.iterrows()):

        to_add = []
        for col, num in row.iteritems():
            if isinstance(num, float):
                if np.isnan(num):
                    to_add.append("")
                else:
                    num = num_format % num
                    to_add.append(math(num))
            else:
                to_add.append(num)
        table.add_row([idx] + to_add)

    table.generate_tex("../08_figures/%s" % save_name)


latex_table(output, "n_en_trading_strat_complexity", num_format="%.3f")
