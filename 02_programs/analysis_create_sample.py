# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %% Packages:
import os

# see https://github.com/bcgsc/mavis/issues/185
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import glob
import time
import argparse
from joblib import dump, Parallel, delayed

from distributed import Client
from dask.distributed import as_completed
from dask_jobqueue import SLURMCluster

from collections import defaultdict

from scoring import scorer
from init_model_parameters import possible_samples


# %% Hyperparameters:
parser = argparse.ArgumentParser()
parser.add_argument(
    "--fileLoc", help="file location. can be in {}".format(possible_samples), required=True
)  # uses data on WWU Cloud.
parser.add_argument("--lin_mod", help="Linear model string, separate by spaces.", required=True)
parser.add_argument("--nlin_mod", help="Nonlinear model string, separate by spaces.", required=True)
args = parser.parse_args()

# file locations
fileLoc = args.fileLoc
saveLoc = fileLoc.replace("04_results", "05_models")
analysisLoc = os.path.join(fileLoc, "analysis")
modelLoc = fileLoc.replace("04_results", "05_models")

# model list
linear_models = args.lin_mod.split(" ")
nonlinear_models = args.nlin_mod.split(" ")

model_types = []
with open(os.path.join(modelLoc, "_model_locations.txt"), "r") as f:
    for line in f:
        model_types.append(line.strip("\n"))

all_models = {}
for model_type in model_types:
    all_models[model_type.split("___")[0] + "_" + model_type.split("___")[-1]] = os.path.join(modelLoc, model_type)

print("\n\n--- Models chosen ---\n")
for m in all_models:
    print(m, flush=True)
print("\n")

print("Linear models:", linear_models, flush=True)
print("Nonlinear models:", nonlinear_models, flush=True)

n_ports = 10


# %% Functions:
def inner_(
    predictions, errors, xs_errors, SHAPs, rel_SHAPs, n_port, class_columns, skip_cols, conditional_SHAPs: dict = {}
):

    # ---- scoring:
    # full sample R2 (weighted by number of observations per month)
    scores = scorer(predictions, skip_cols)
    scores = pd.Series(
        scores,
        index=[c for c in predictions.columns if c not in skip_cols],
    )
    scores.name = "scores"

    # monthly scoring:
    monthly_scores = predictions.groupby("date").apply(lambda x: scorer(x, skip_cols)).apply(pd.Series)
    monthly_scores.columns = [c for c in predictions.columns if c not in skip_cols]

    # annual scoring:
    annual_scores = predictions.groupby(pd.Grouper(freq="A")).apply(lambda x: scorer(x, skip_cols)).apply(pd.Series)
    annual_scores.columns = [c for c in predictions.columns if c not in skip_cols]

    # expected return portfolios:
    predictions["port"] = predictions.groupby("date")["predicted"].transform(
        lambda x: pd.qcut(x, n_port, labels=False, duplicates="drop")
    )
    predictions["port"] = predictions["port"].astype("i4")

    # classification group scoring:
    class_scores = {}
    for c in class_columns:
        print("Class: {}.".format(c))
        tmp = predictions.groupby(c).apply(lambda x: scorer(x, skip_cols)).apply(pd.Series)
        tmp.columns = ["score"]
        class_scores[c] = tmp.copy()

    return_dict = {
        "scores": scores.copy(),
        "monthly_scores": monthly_scores.copy(),
        "annual_scores": annual_scores.copy(),
        "class_scores": class_scores.copy(),
        "predictions": predictions[[c for c in predictions.columns if c not in class_columns]].copy(),
        "errors": errors.copy(),
        "xs_errors": xs_errors.copy(),
        "SHAPs": SHAPs.copy(),
        "rel_SHAPs": rel_SHAPs.copy(),
        "conditional_SHAPs": conditional_SHAPs.copy(),
    }
    return return_dict


def get_shaps(shaps, skip_cols, rel: bool = False):
    out = shaps[[c for c in shaps.columns if c not in skip_cols]]
    if rel:
        out = out.groupby("date").mean()
    else:
        out = out.abs().groupby("date").mean()
    out["N"] = shaps.groupby("date").size()
    return out


def get_errors(pred, skip_cols):
    target = pred.target
    N = pred.groupby("date").size()
    pred = pred[[c for c in pred.columns if c not in skip_cols]]
    error = (pred.sub(target, axis=0)).pow(2).groupby("date").sum()
    error["target"] = target.pow(2).groupby("date").sum()
    error["N"] = N
    return error


def get_xs_errors(pred, skip_cols):
    target = pred.target - pred.target.groupby("date").transform("mean")
    N = pred.groupby("date").size()
    pred = pred[[c for c in pred.columns if c not in skip_cols]]
    pred = pred - pred.groupby("date").transform("mean")
    error = (pred.sub(target, axis=0)).pow(2).groupby("date").sum()
    error["target"] = target.pow(2).groupby("date").sum()
    error["N"] = N
    return error


def create_ensemble_predictions_fast_inner(i_file: int, models, model_name: str):
    errors = {}
    xs_errors = {}
    SHAPs = {}
    rel_SHAPs = {}
    conditional_SHAPs = {}

    class_groups = pd.read_parquet(os.path.join(analysisLoc, "class_groups.pq"))
    class_groups = class_groups.drop(columns=[c for c in class_groups.columns if "return_" in c or "gain_" in c])
    class_groups["date"] = class_groups["date"].dt.to_timestamp()
    class_columns = class_groups.columns.tolist()
    class_columns = [c for c in class_columns if c not in ["date", "optionid", "permno"]]
    skip_cols = ["target", "optionid", "date", "port", "permno"] + class_columns

    per_file_ensemble = []
    per_file_shap = []
    for i_model, model in enumerate(models):
        # print(model, flush=True)
        predictionLoc = all_models[[key for key in all_models.keys() if key.rsplit("_", 1)[0] == model][0]]
        file = os.path.join(predictionLoc, "predictions_%d.pq" % i_file)
        # print("Reading file %s" % file, flush=True)

        pred = pd.read_parquet(file)
        pred.columns = [c.replace(":", "-") for c in pred.columns]  # needed due to change in sample strings.
        # pred["date"] = pred["date"].dt.to_period("M")
        pred = pred.set_index(["date", "optionid"])
        pred = pred.sort_index()  # make sure that every value is sorted
        if len(per_file_ensemble) == 0:
            per_file_ensemble = pred.copy()
        else:
            per_file_ensemble = (per_file_ensemble * i_model + pred.copy()) / (i_model + 1)

        # SHAPs
        shaps = pd.read_parquet(file.replace("predictions_", "shaps_"))
        group_shaps = pd.read_parquet(file.replace("predictions_", "group_shaps_"))
        shaps = pd.concat((shaps, group_shaps), axis=1)
        shaps = shaps.reset_index()
        # shaps["date"] = shaps["date"].dt.to_period("M")
        shaps = shaps.set_index(["date", "optionid"])
        shaps = shaps.sort_index()  # make sure that every value is sorted
        if len(per_file_shap) == 0:
            per_file_shap = shaps.copy()
        else:
            per_file_shap = (per_file_shap * i_model + shaps.copy()) / (i_model + 1)

    per_file_ensemble = per_file_ensemble.reset_index()
    per_file_shap = per_file_shap.reset_index()
    print(per_file_ensemble.shape, flush=True)
    print(per_file_shap.shape, flush=True)

    # full sample errors:
    per_file_ensemble = per_file_ensemble.merge(class_groups, on=["date", "optionid"])
    per_file_ensemble = per_file_ensemble.set_index("date")
    errors["full"] = get_errors(per_file_ensemble, skip_cols)
    xs_errors["full"] = get_xs_errors(per_file_ensemble, skip_cols)

    # SHAPs:

    # use SHAPs to show partial dependence plots, both in a univariate and bivariate
    # (independent double sorts) setting.
    per_file_shap = per_file_shap.set_index(["date", "optionid"])

    # obtain characteristics for given year:
    year = per_file_shap.index.get_level_values("date").year.unique()[0]
    char_files = glob.glob(os.path.join(fileLoc, "*%d.pq" % year))
    char_files = [f for f in char_files if "classifier_" not in f]
    chars = pd.concat([pd.read_parquet(f) for f in char_files])
    chars = chars.reset_index()
    # chars["date"] = chars["date"].dt.to_period("M")
    chars = chars.set_index(["date", "optionid"])
    cond_vars = (
        chars[["moneyness", "ttm"]]
        .groupby("date")
        .apply(lambda x: x.apply(lambda y: pd.qcut(y, q=5, labels=False, duplicates="drop")))
    ) + 1

    char_q = (
        chars[[c for c in chars.columns if c in per_file_shap.columns]]
        .groupby("date")
        .apply(lambda x: x.apply(lambda y: pd.qcut(y, q=100, labels=False, duplicates="drop")))
    )  # contains percentiles
    # chars /= chars.groupby("date").max()

    cond_single_all = []
    cond_mon_all = []
    cond_ttm_all = []
    cond_bucket_all = []
    for char in char_q.columns:
        print(char, flush=True)
        shap_tmp = per_file_shap[[char]]
        char_tmp = char_q[[char]]
        char_tmp.columns = ["percentile"]
        shap_tmp = shap_tmp.merge(char_tmp, on=["date", "optionid"])
        cond_single = shap_tmp.groupby(["date", "percentile"]).mean()
        cond_single_all.append(cond_single)

        if char not in ["moneyness", "ttm"]:
            # moneyness conditional sort
            mon = cond_vars["moneyness"]
            shap_tmp = shap_tmp.merge(mon, on=["date", "optionid"])
            cond_mon = shap_tmp.groupby(["date", "percentile", "moneyness"])[char].mean()
            cond_mon_all.append(cond_mon)
            # ttm conditional sort
            ttm = cond_vars["ttm"]
            shap_tmp = shap_tmp.merge(ttm, on=["date", "optionid"])
            cond_ttm = shap_tmp.groupby(["date", "percentile", "ttm"])[char].mean()
            cond_ttm_all.append(cond_ttm)

        # bucket stuff
        shap_tmp = shap_tmp.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])
        print(shap_tmp.groupby("bucket").size(), flush=True)
        cond_bucket = shap_tmp.groupby(["date", "percentile", "bucket"])[char].mean()
        cond_bucket_all.append(cond_bucket)

    conditional_SHAPs["single"] = pd.concat(cond_single_all, axis=1)
    conditional_SHAPs["moneyness"] = pd.concat(cond_mon_all, axis=1)
    conditional_SHAPs["ttm"] = pd.concat(cond_ttm_all, axis=1)
    conditional_SHAPs["bucket"] = pd.concat(cond_bucket_all, axis=1)

    # standard absolute and relative SHAP (average per month):
    per_file_shap = per_file_shap.reset_index()
    per_file_shap = per_file_shap.merge(class_groups, on=["date", "optionid"])
    per_file_shap = per_file_shap.set_index("date")
    SHAPs["full"] = get_shaps(per_file_shap, skip_cols)
    rel_SHAPs["full"] = get_shaps(per_file_shap, skip_cols, rel=True)

    # class group errors:
    for c in class_columns:
        print("Errors for class: {}.".format(c), flush=True)
        errors[c] = per_file_ensemble.groupby(c).apply(lambda x: get_errors(x, skip_cols))
        xs_errors[c] = per_file_ensemble.groupby(c).apply(lambda x: get_xs_errors(x, skip_cols))
        SHAPs[c] = per_file_shap.groupby(c).apply(lambda x: get_shaps(x, skip_cols))
        rel_SHAPs[c] = per_file_shap.groupby(c).apply(lambda x: get_shaps(x, skip_cols, rel=True))

    per_file_ensemble = per_file_ensemble.drop(
        columns=[c for c in per_file_ensemble.columns if c not in skip_cols + ["predicted", "target"]]
    )

    return model_name, per_file_ensemble, errors, xs_errors, SHAPs, rel_SHAPs, conditional_SHAPs


def create_ensemble_predictions_fast(
    predictions, errors, xs_errors, SHAPs, rel_SHAPs, conditional_SHAPs, model_name: str
):
    class_groups = pd.read_parquet(os.path.join(analysisLoc, "class_groups.pq"))
    class_groups = class_groups.drop(columns=[c for c in class_groups.columns if "return_" in c or "gain_" in c])
    class_groups["date"] = class_groups["date"].dt.to_timestamp()
    class_columns = class_groups.columns.tolist()
    class_columns = [c for c in class_columns if c not in ["date", "optionid", "permno"]]
    skip_cols = ["target", "optionid", "date", "port", "permno"] + class_columns

    predictions = pd.concat(predictions).sort_index()
    for key in errors:
        errors[key] = pd.concat(errors[key])
        errors[key] = errors[key].sort_index()
    for key in xs_errors:
        xs_errors[key] = pd.concat(xs_errors[key])
        xs_errors[key] = xs_errors[key].sort_index()
    for key in SHAPs:
        SHAPs[key] = pd.concat(SHAPs[key])
        SHAPs[key] = SHAPs[key].sort_index()
    for key in rel_SHAPs:
        rel_SHAPs[key] = pd.concat(rel_SHAPs[key])
        rel_SHAPs[key] = rel_SHAPs[key].sort_index()
    for key in conditional_SHAPs:
        conditional_SHAPs[key] = pd.concat(conditional_SHAPs[key]).sort_index()

    return model_name, inner_(
        predictions,
        errors,
        xs_errors,
        SHAPs,
        rel_SHAPs,
        n_ports,
        class_columns,
        skip_cols,
        conditional_SHAPs=conditional_SHAPs,
    )


def create_ensemble_predictions(N_files, models, all_models, model_name: str):
    print(model_name, flush=True)
    class_groups = pd.read_parquet(os.path.join(analysisLoc, "class_groups.pq"))
    class_groups = class_groups.drop(columns=[c for c in class_groups.columns if "return_" in c or "gain_" in c])
    class_groups["date"] = class_groups["date"].dt.to_timestamp()
    class_columns = class_groups.columns.tolist()
    class_columns = [c for c in class_columns if c not in ["date", "optionid", "permno"]]
    skip_cols = ["target", "optionid", "date", "port", "permno"] + class_columns

    predictions = []
    errors = defaultdict(list)
    xs_errors = defaultdict(list)
    SHAPs = defaultdict(list)
    rel_SHAPs = defaultdict(list)
    conditional_SHAPs = defaultdict(list)
    for i_file in range(N_files):
        print("%d/%d" % (i_file + 1, N_files), flush=True)
        per_file_ensemble = []
        per_file_shap = []
        for i_model, model in enumerate(models):
            print(model)
            predictionLoc = all_models[[key for key in all_models.keys() if key.rsplit("_", 1)[0] == model][0]]
            file = os.path.join(predictionLoc, "predictions_%d.pq" % i_file)

            pred = pd.read_parquet(file)
            pred.columns = [c.replace(":", "-") for c in pred.columns]  # needed due to change in sample strings.
            # pred["date"] = pred["date"].dt.to_period("M")
            pred = pred.set_index(["date", "optionid"])
            pred = pred.sort_index()  # make sure that every value is sorted
            print(pred, flush=True)
            if len(per_file_ensemble) == 0:
                per_file_ensemble = pred.copy()
            else:
                per_file_ensemble = (per_file_ensemble * i_model + pred.copy()) / (i_model + 1)

            # SHAPs
            shaps = pd.read_parquet(file.replace("predictions_", "shaps_"))
            group_shaps = pd.read_parquet(file.replace("predictions_", "group_shaps_"))
            shaps = pd.concat((shaps, group_shaps), axis=1)
            shaps = shaps.reset_index()
            # shaps["date"] = shaps["date"].dt.to_period("M")
            shaps = shaps.set_index(["date", "optionid"])
            shaps = shaps.sort_index()  # make sure that every value is sorted
            print(shaps, flush=True)
            if len(per_file_shap) == 0:
                per_file_shap = shaps.copy()
            else:
                per_file_shap = (per_file_shap * i_model + shaps.copy()) / (i_model + 1)

        per_file_ensemble = per_file_ensemble.reset_index()
        per_file_shap = per_file_shap.reset_index()

        # full sample errors:
        per_file_ensemble = per_file_ensemble.merge(class_groups, on=["date", "optionid"])
        per_file_ensemble = per_file_ensemble.set_index("date")
        errors["full"].append(get_errors(per_file_ensemble, skip_cols))
        xs_errors["full"].append(get_xs_errors(per_file_ensemble, skip_cols))

        # SHAPs:

        # use SHAPs to show partial dependence plots, both in a univariate and bivariate
        # (independent double sorts) setting.
        per_file_shap = per_file_shap.set_index(["date", "optionid"])

        # obtain characteristics for given year:
        year = per_file_shap.index.get_level_values("date").year.unique()[0]
        char_files = glob.glob(os.path.join(fileLoc, "*%d.pq" % year))
        char_files = [f for f in char_files if "classifier_" not in f]
        chars = pd.concat([pd.read_parquet(f) for f in char_files])
        chars = chars.reset_index()
        # chars["date"] = chars["date"].dt.to_period("M")
        chars = chars.set_index(["date", "optionid"])
        cond_vars = (
            chars[["moneyness", "ttm"]]
            .groupby("date")
            .apply(lambda x: x.apply(lambda y: pd.qcut(y, q=5, labels=False, duplicates="drop")))
        ) + 1

        char_q = (
            chars[[c for c in chars.columns if c in per_file_shap.columns]]
            .groupby("date")
            .apply(lambda x: x.apply(lambda y: pd.qcut(y, q=100, labels=False, duplicates="drop")))
        )  # contains percentiles
        # chars /= chars.groupby("date").max()

        def cond_shaps(char):
            shap_tmp = per_file_shap[[char]]
            char_tmp = char_q[[char]]
            char_tmp.columns = ["percentile"]
            shap_tmp = shap_tmp.merge(char_tmp, on=["date", "optionid"])
            cond_single = shap_tmp.groupby(["date", "percentile"]).mean()

            if char not in ["moneyness", "ttm"]:
                # moneyness conditional sort
                mon = cond_vars["moneyness"]
                shap_tmp = shap_tmp.merge(mon, on=["date", "optionid"])
                cond_mon = shap_tmp.groupby(["date", "percentile", "moneyness"])[char].mean()
                # ttm conditional sort
                ttm = cond_vars["ttm"]
                shap_tmp = shap_tmp.merge(ttm, on=["date", "optionid"])
                cond_ttm = shap_tmp.groupby(["date", "percentile", "ttm"])[char].mean()
            else:
                cond_mon = None
                cond_ttm = None

            # bucket stuff
            shap_tmp = shap_tmp.merge(class_groups[["date", "optionid", "bucket"]], on=["date", "optionid"])
            cond_bucket = shap_tmp.groupby(["date", "percentile", "bucket"])[char].mean()

            return cond_single, cond_mon, cond_ttm, cond_bucket

        conditional_SHAPs["single"] = [pd.DataFrame()]

        # standard absolute and relative SHAP (average per month):
        per_file_shap = per_file_shap.reset_index()
        per_file_shap = per_file_shap.merge(class_groups, on=["date", "optionid"])
        per_file_shap = per_file_shap.set_index("date")
        SHAPs["full"].append(get_shaps(per_file_shap, skip_cols))
        rel_SHAPs["full"].append(get_shaps(per_file_shap, skip_cols, rel=True))

        # class group errors:
        for c in class_columns:
            print("Errors for class: {}.".format(c), flush=True)
            errors[c].append(per_file_ensemble.groupby(c).apply(lambda x: get_errors(x, skip_cols)))
            xs_errors[c].append(per_file_ensemble.groupby(c).apply(lambda x: get_xs_errors(x, skip_cols)))
            SHAPs[c].append(per_file_shap.groupby(c).apply(lambda x: get_shaps(x, skip_cols)))
            rel_SHAPs[c].append(per_file_shap.groupby(c).apply(lambda x: get_shaps(x, skip_cols, rel=True)))

        per_file_ensemble = per_file_ensemble.drop(
            columns=[c for c in per_file_ensemble.columns if c not in skip_cols + ["predicted", "target"]]
        )
        predictions.append(per_file_ensemble)

    predictions = pd.concat(predictions).sort_index()
    for key in errors:
        errors[key] = pd.concat(errors[key])
        errors[key] = errors[key].sort_index()
    for key in xs_errors:
        xs_errors[key] = pd.concat(xs_errors[key])
        xs_errors[key] = xs_errors[key].sort_index()
    for key in SHAPs:
        SHAPs[key] = pd.concat(SHAPs[key])
        SHAPs[key] = SHAPs[key].sort_index()
    for key in rel_SHAPs:
        rel_SHAPs[key] = pd.concat(rel_SHAPs[key])
        rel_SHAPs[key] = rel_SHAPs[key].sort_index()
    for key in conditional_SHAPs:
        conditional_SHAPs[key] = pd.concat(conditional_SHAPs[key]).sort_index()

    return model_name, inner_(
        predictions,
        errors,
        xs_errors,
        SHAPs,
        rel_SHAPs,
        n_ports,
        class_columns,
        skip_cols,
        conditional_SHAPs=conditional_SHAPs,
    )


def create_model_predictions(predictionLoc, model_name: str):
    print(model_name, flush=True)
    class_groups = pd.read_parquet(os.path.join(analysisLoc, "class_groups.pq"))
    class_groups = class_groups.drop(columns=[c for c in class_groups.columns if "return_" in c or "gain_" in c])
    class_groups["date"] = class_groups["date"].dt.to_timestamp()
    class_columns = class_groups.columns.tolist()
    class_columns = [c for c in class_columns if c not in ["date", "optionid", "permno"]]
    skip_cols = ["target", "optionid", "date", "port", "permno"] + class_columns

    print("\n\n\n\n--- %s --- \n" % predictionLoc)
    prediction_files = glob.glob(os.path.join(predictionLoc, "predictions*"))

    # ---- raw predictions and errors per month:
    predictions = []
    errors = defaultdict(list)
    xs_errors = defaultdict(list)
    SHAPs = defaultdict(list)
    rel_SHAPs = defaultdict(list)
    for i_file, _ in enumerate(prediction_files):
        print("%d/%d" % (i_file, len(prediction_files)), flush=True)
        file = os.path.join(predictionLoc, "predictions_%d.pq" % i_file)

        pred = pd.read_parquet(file)
        pred.columns = [c.replace(":", "-") for c in pred.columns]  # needed due to change in sample strings.
        # pred["date"] = pred["date"].dt.to_period("M")
        pred = pred.set_index(["date", "optionid"])
        pred = pred.sort_index()  # make sure that every value is sorted
        print(pred, flush=True)
        pred = pred.merge(class_groups, on=["date", "optionid"])
        pred = pred.set_index("date")

        # ---- SHAPs
        shaps = pd.read_parquet(file.replace("predictions_", "shaps_"))
        group_shaps = pd.read_parquet(file.replace("predictions_", "group_shaps_"))

        # absolute and relative SHAPs per year
        shaps = pd.concat((shaps, group_shaps), axis=1)
        shaps = shaps.reset_index()
        # shaps["date"] = shaps["date"].dt.to_period("M")
        shaps = shaps.set_index(["date", "optionid"])
        shaps = shaps.sort_index()  # make sure that every value is sorted
        print(shaps, flush=True)
        shaps = shaps.merge(class_groups, on=["date", "optionid"])
        shaps = shaps.set_index("date")
        SHAPs["full"].append(get_shaps(shaps, skip_cols))
        rel_SHAPs["full"].append(get_shaps(shaps, skip_cols, rel=True))

        # full sample errors
        errors["full"].append(get_errors(pred, skip_cols))
        xs_errors["full"].append(get_xs_errors(pred, skip_cols))

        # class group errors:
        for c in class_columns:
            print("Errors for class: {}.".format(c), flush=True)
            errors[c].append(pred.groupby(c).apply(lambda x: get_errors(x, skip_cols)))
            xs_errors[c].append(pred.groupby(c).apply(lambda x: get_xs_errors(x, skip_cols)))
            SHAPs[c].append(shaps.groupby(c).apply(lambda x: get_shaps(x, skip_cols)))
            rel_SHAPs[c].append(shaps.groupby(c).apply(lambda x: get_shaps(x, skip_cols, rel=True)))

        pred = pred.drop(columns=[c for c in pred.columns if c not in skip_cols + ["predicted", "target"]])
        predictions.append(pred)

    # concat values:
    print("Concat", flush=True)
    predictions = pd.concat(predictions)
    predictions = predictions.sort_index()
    for key in errors:
        errors[key] = pd.concat(errors[key])
        errors[key] = errors[key].sort_index()
    for key in xs_errors:
        xs_errors[key] = pd.concat(xs_errors[key])
        xs_errors[key] = xs_errors[key].sort_index()
    for key in SHAPs:
        SHAPs[key] = pd.concat(SHAPs[key])
        SHAPs[key] = SHAPs[key].sort_index()
    for key in rel_SHAPs:
        rel_SHAPs[key] = pd.concat(rel_SHAPs[key])
        rel_SHAPs[key] = rel_SHAPs[key].sort_index()

    return model_name, inner_(predictions, errors, xs_errors, SHAPs, rel_SHAPs, n_ports, class_columns, skip_cols)


# %%
if __name__ == "__main__":
    target_years = np.arange(1996, 2020 + 1)
    # ---- set up dask clusters
    print("Starting SLURM cluster.", flush=True)
    os.makedirs("../logs/analysis/", exist_ok=True)
    logFile = "../logs/analysis/%j.%N.out"
    [os.remove(f) for f in glob.glob("../logs/analysis/*")]
    MEM = int(60 * 1024)
    CORES = 32
    cluster = SLURMCluster(
        queue="normal,largesmp,bigsmp",
        cores=1,
        processes=1,
        memory=str(MEM) + "MB",
        walltime="12:00:00",
        local_directory="/scratch/tmp/h_beck18/",
        extra=["--resources memory=1"],
        job_extra=[
            "-J options",
            "-c " + str(CORES),
            "-o " + logFile,
            "-e " + logFile.replace(".out", ".err"),
        ],
    )
    cluster.adapt(minimum=1, maximum=11 + 18 * 2, target_duration="1s")

    time.sleep(10)
    client = Client(cluster)

    # ---- run computations
    start = time.time()

    print(client)
    print(cluster)
    # time.sleep(30)

    key_dict = {
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "ElasticNet": "ENet",
        "PCR": "PCR",
        "PLS": "PLS",
        "linear": "L-En",
        "LightGBR": "GBR",
        "LightRF": "RF",
        "LightDart": "Dart",
        "FFNN": "FFN",
        "nonlinear": "N-En",
    }

    prediction_dict = {}

    # ---- ensembles:
    N_files = len(glob.glob(os.path.join(list(all_models.values())[0], "predictions*")))

    df_futures = [
        client.submit(
            create_model_predictions,
            predictionLoc=all_models[model],
            model_name=[key_dict[key] for key in key_dict if key in model][0],  # model name.
            resources={"memory": 1},
        )
        for model in all_models
    ]
    df_futures.append(
        client.submit(
            create_ensemble_predictions,
            N_files=N_files,
            models=linear_models,
            all_models=all_models,
            model_name="L-En",
            resources={"memory": 1},
        )
    )
    df_futures.append(
        client.submit(
            create_ensemble_predictions,
            N_files=N_files,
            models=nonlinear_models,
            all_models=all_models,
            model_name="N-En",
            resources={"memory": 1},
        )
    )
    seq = as_completed(df_futures, with_results=False)
    for done_work in seq:
        try:
            print("Job done.")
            model, result = done_work.result()
            prediction_dict[model] = result
        except Exception as error:
            print("Error in...", flush=True)
            print(done_work, flush=True)
            print(error, flush=True)

    # rearrange.
    prediction_dict = {key: prediction_dict[key] for key in key_dict.values()}
    dump(prediction_dict, os.path.join(saveLoc, "prediction_dict_comparison.pkl"))
    print("Done.")


# %%
