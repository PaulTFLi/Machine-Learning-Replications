# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm


# %%
def oos_r2(target, predicted):
    return 1 - np.sum((target.reshape(-1, 1) - predicted.reshape(-1, 1)) ** 2) / np.sum(target**2)


def mse(target, predicted):
    return np.mean((target.reshape(-1, 1) - predicted.reshape(-1, 1)) ** 2)


# %%
def predictive_fit(X, y, model):
    r2 = oos_r2(y.to_numpy(), model.predict(X))
    return r2


def scorer(df, skip_cols=[]):
    scores = []
    for col in df.columns:
        if col in skip_cols:
            continue
        scores.append(oos_r2(df["target"].to_numpy(), df[col].to_numpy()))
    return scores


# %% Diebold/Mariano and Forecast Correlation and Clark and West (2007):
def DieboldMariano(predictions, HAC_lags):
    cols = [c for c in predictions.columns if c != "target"]
    diebold_test = pd.DataFrame(columns=cols, index=cols)
    for i, col in enumerate(predictions.columns):
        for mod in predictions.columns[i:]:
            if col != "target" and mod != "target":
                if mod != col and col != "target" and mod != "target":
                    tmp = (
                        (
                            predictions[mod].sub(predictions["target"]).pow(2)
                            - predictions[col].sub(predictions["target"]).pow(2)
                        )
                        .groupby("date")
                        .mean()
                        .to_frame()
                    )
                    tmp.columns = [mod]
                    se = (
                        sm.OLS(tmp[mod], np.ones(tmp.shape[0]))
                        .fit(cov_type="HAC", cov_kwds={"maxlags": HAC_lags})
                        .bse.values[0]
                    )
                    diebold_test.loc[col, mod] = tmp.mean().values[0] / se
                else:
                    diebold_test.loc[col, mod] = np.nan
    return (diebold_test.astype("f4") * (-1)).iloc[:-1, 1:]


def DieboldMariano_XS(predictions, HAC_lags):
    cols = [c for c in predictions.columns if c != "target"]
    diebold_test = pd.DataFrame(columns=cols, index=cols)
    for i, col in enumerate(predictions.columns):
        for mod in predictions.columns[i:]:
            if col != "target" and mod != "target":
                if mod != col and col != "target" and mod != "target":
                    tmp = (
                        (
                            (
                                (predictions[mod] - predictions[mod].groupby("date").transform("mean"))
                                - (predictions["target"] - predictions["target"].groupby("date").transform("mean"))
                            ).pow(2)
                            - (
                                (predictions[col] - predictions[col].groupby("date").transform("mean"))
                                - (predictions["target"] - predictions["target"].groupby("date").transform("mean"))
                            ).pow(2)
                        )
                        .groupby("date")
                        .mean()
                        .to_frame()
                    )
                    tmp.columns = [mod]
                    se = (
                        sm.OLS(tmp[mod], np.ones(tmp.shape[0]))
                        .fit(cov_type="HAC", cov_kwds={"maxlags": HAC_lags})
                        .bse.values[0]
                    )
                    diebold_test.loc[col, mod] = tmp.mean().values[0] / se
                else:
                    diebold_test.loc[col, mod] = np.nan
    return (diebold_test.astype("f4") * (-1)).iloc[:-1, 1:]


def ForecastCorrelation(predictions):
    forecast_correlation = predictions.corr()
    for i, col in enumerate(predictions.columns):
        for mod in predictions.columns[: (i + 1)]:
            forecast_correlation.loc[col, mod] = np.nan
    return forecast_correlation.iloc[:-1, 1:]


def TS_R2(predictions):
    cols = [c for c in predictions.columns if c != "target"]
    xs_r2 = pd.Series(index=cols, dtype="float64")
    for mod in cols:
        num = (((predictions["target"] - predictions[mod])).pow(2)).sum()
        denom = predictions["target"].pow(2).sum()
        xs_r2.loc[mod] = 1 - num / denom
    return xs_r2.astype("f4")


def XS_R2(predictions):
    cols = [c for c in predictions.columns if c != "target"]
    xs_r2 = pd.Series(index=cols, dtype="float64")
    for mod in cols:
        num = (
            (
                (predictions["target"] - predictions["target"].groupby("date").transform("mean"))
                - (predictions[mod] - predictions[mod].groupby("date").transform("mean"))
            ).pow(2)
        ).sum()
        denom = (predictions["target"] - predictions["target"].groupby("date").transform("mean")).pow(2).sum()
        xs_r2.loc[mod] = 1 - num / denom
    return xs_r2.astype("f4")


def ClarkWest(predictions, HAC_lags, benchmark_type: str = "zero", cw_adjust: bool = True):
    cols = [c for c in predictions.columns if c != "target"]
    cw_test = pd.Series(index=cols, dtype="float64")
    for mod in cols:
        if benchmark_type == "zero":
            if cw_adjust:
                tmp = (
                    (
                        predictions["target"].pow(2)
                        - predictions[mod].sub(predictions["target"]).pow(2)
                        + predictions[mod].pow(2)
                    )
                    .groupby("date")
                    .mean()
                    .to_frame()
                )
            else:
                tmp = (
                    (predictions["target"].pow(2) - predictions[mod].sub(predictions["target"]).pow(2))
                    .groupby("date")
                    .mean()
                    .to_frame()
                )
        elif benchmark_type == "mean":
            if cw_adjust:
                tmp = (
                    (
                        (predictions["target"].groupby("date").mean().shift() - predictions["target"]).pow(2)
                        - predictions[mod].sub(predictions["target"]).pow(2)
                        + (predictions["target"].mean() - predictions[mod]).pow(2)
                    )
                    .groupby("date")
                    .mean()
                    .to_frame()
                )
            else:
                tmp = (
                    (
                        (predictions["target"].groupby("date").mean().shift() - predictions["target"]).pow(2)
                        - predictions[mod].sub(predictions["target"]).pow(2)
                    )
                    .groupby("date")
                    .mean()
                    .to_frame()
                )
        elif benchmark_type == "xs":
            tmp = (
                (
                    (predictions["target"] - predictions["target"].groupby("date").transform("mean")).pow(2)
                    - (
                        (predictions["target"] - predictions["target"].groupby("date").transform("mean"))
                        - (predictions[mod] - predictions[mod].groupby("date").transform("mean"))
                    ).pow(2)
                )
                .groupby("date")
                .mean()
                .to_frame()
            )
        tmp.columns = [mod]
        se = (
            sm.OLS(tmp[mod], np.ones(tmp.shape[0]), missing="drop")
            .fit(cov_type="HAC", cov_kwds={"maxlags": HAC_lags})
            .bse.values[0]
        )
        cw_test.loc[mod] = tmp.mean().values[0] / se
    return cw_test.astype("f4")


# %%
def feature_importance(
    X,
    model,
    predictor_func,
    char_names: list,
    groups: dict = {},
    group_prefix: str = "f",
    exclude: list = ["ind_"],
    verbose=False,
):
    pd.options.mode.chained_assignment = None

    if not groups:
        char_names = [c.rsplit("_", 1)[0] for c in char_names]
        if len(groups) == 0:
            groups = {c: [c] for c in char_names if c not in exclude}
        char_names = pd.Index(char_names)

    def inner_(g, X):
        if verbose:
            print("Working on group {}.".format(g))
            print("Members: \n{}\n============.".format(groups[g]))
        ii = char_names.isin(groups[g])
        old_values = X[:, ii].copy()
        X[:, ii] = 0
        predicted = predictor_func(X, model)
        X[:, ii] = old_values
        return predicted.flatten()

    feature_scores = {group_prefix + "_" + g: inner_(g, X) for g in list(groups.keys())}

    return feature_scores


# ---- https://www.kaggle.com/estevaouyra/shap-advanced-uses-grouping-and-correlation
def grouped_shap(shap_df, groups):
    from itertools import repeat, chain

    def revert_dict(d):
        return dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))

    groupmap = revert_dict(groups)
    shap_df = shap_df.T
    shap_df["group"] = shap_df.reset_index().features.map(groupmap).values
    shap_grouped = shap_df.groupby("group").sum().T
    return shap_grouped


def SHAP_importance(shap_vals, char_names: list, groups: dict = {}, group_prefix: str = "f"):
    pd.options.mode.chained_assignment = None

    # if not groups:  # option to exclude some features (industry dummies, for example):
    #     char_names = [c.rsplit("_", 1)[0] for c in char_names]
    #     if len(groups) == 0:
    #         groups = {c: [c] for c in char_names if c not in exclude}
    #     char_names = pd.Index(char_names)

    # exp = SHAP_explainer(model)
    # shap_vals = SHAP_explainer.shap_values(X)  # test inputs
    shap_df = pd.DataFrame(shap_vals, columns=pd.Index(char_names, name="features"))

    if groups:
        group_shap_df = grouped_shap(shap_df, groups)
        group_shap_df.columns = [group_prefix + "_" + g for g in group_shap_df.columns]

    return shap_df, group_shap_df


# %%
