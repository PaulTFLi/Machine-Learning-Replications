# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
""" Predict.py

    Creates predictions of fitted models (using hyperparameter_optimization.py).
    Includes the following features:
        - Ensemble predictions, using --n_ensemble models
        - Feature importance of single features
        - Feature importance of groups, specified in `features.xlsx`

Raises:
    ValueError: Wrong save type selected.
    IndexError: Config for model step is unavailable.

Returns:
    None: saves .pq files of predictions and feature importance scores.
"""


# %%
from multiprocessing import cpu_count
import os
import socket
import glob
import pandas as pd
import numpy as np
import time
import argparse

from joblib import load

from init_model_parameters import (
    init_models,
    possible_models,
    possible_samples,
)
from scoring import SHAP_importance, feature_importance
from data_loading import read_data_single_file

from joblib import Parallel, delayed

import torch
import torch.nn as nn

import lightgbm as lgbm

import shap

import ray


# %% Hyperparameters
use_refit = False


host = socket.gethostname()
if "D-1210" in host:
    print("Local.\n")
    LOCAL = True

    print("Possible samples:")
    for i, elem in enumerate(possible_samples):
        print("%d: %s" % (i, elem))
    fileLoc = input("Which file location?   ")
    fileLoc = possible_samples[int(fileLoc)]
    modelLoc = fileLoc.replace("04_results", "05_models")

    print("Possible models for this file:")
    for i, (model_step, s) in enumerate(possible_models.items()):
        print("%d: %s (library %s)" % (i, model_step, s))
    print("\n\n")
    model_type = input("Which model?   ")
    model_type = list(possible_models.keys())[int(model_type)]
    library = possible_models[model_type]

    model_files = glob.glob(os.path.join(modelLoc, model_type + "___") + "*")
    model_files.sort()

    if len(model_files) > 1:
        print("Choose the respective validated set of models:")
        for i, f in enumerate(model_files):
            print("%d: %s" % (i, os.path.split(f)[-1]))
        use_file = input("Choice:   ")
    else:
        use_file = 0
    model_files = model_files[int(use_file)]

    # obtain the time stamp of the trial + results_table
    results_table = pd.read_pickle(os.path.join(model_files, model_type + "_results_table.pkl"))
    for ith_step in results_table.index:
        tmp = results_table.loc[ith_step, "test_files"]
        tmp = [os.path.join(fileLoc, t.split(fileLoc[2:])[1]) for t in tmp]
        results_table.at[ith_step, "test_files"] = tmp

else:
    print("Remote.\n")
    LOCAL = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--experimentLoc", help="Location of model files.")
    parser.add_argument("--model", help="model_type to use. For a list, check 'possible_models'.")
    parser.add_argument("--fileLoc", help="file location. can be in {}".format(possible_samples))
    parser.add_argument("--n_ensemble", help="Number of models in ensemble.", type=int)
    parser.add_argument("--ith_step", help="model_type to use. For a list, check 'possible_models'.")
    parser.add_argument(
        "--feature_importance_predictions",
        help="Whether to return predictions by feature importance, or simply aggregate R2.",
        action="store_true",
    )
    args = parser.parse_args()
    ith_step = int(args.ith_step)
    model_files = args.experimentLoc
    model_type = args.model
    fileLoc = args.fileLoc
    modelLoc = fileLoc.replace("04_results", "05_models")
    n_ensemble = args.n_ensemble
    feature_importance_predictions = args.feature_importance_predictions

    library = possible_models[model_type]
    results_table = pd.read_pickle(os.path.join(model_files, model_type + "_results_table.pkl"))


info_dict = load(os.path.join(model_files, model_type + "_info_dict"))
timestamp = info_dict["trial_datetime"]
target_column = info_dict["target_column"]
date_column = info_dict["date_column"]
other_columns_to_delete = info_dict["other_columns_to_delete"]
other_columns_to_delete.remove("optionid")
C = info_dict["C"]
sample_type = info_dict["feature_group"]

# ---- get number of cpus/gpus for estimation:
# NOTE: depends on size of current data set + number of features selected.
N = pd.read_parquet(results_table["test_files"].iloc[0][-1], columns=["date_col"]).shape[0]
C = info_dict["C"]
mem_required = (N * C) * 4  # bytes needed
mem_required /= 1_000_000_000  # GB needed
mem_required *= 4  # buffer for predictions/errors in memory.
if library == "lightgbm":  # smaller binary files.
    mem_required /= 4

cpus = int(max(8, 72 / (90 / mem_required)))  # some safeguarding against excessively large datasets (max cores = 72).
possible_cpus = np.array([8, 9, 12, 18, 24, 36, 72, 100000])
cpus = int(possible_cpus[np.where(cpus <= possible_cpus)[0][0]])
cpus = min(72, cpus)

gpus = 0
use_cuda = False
if library == "pytorch":
    # if gpus available, request no cpus:
    if torch.cuda.is_available():
        print("Using CUDA.")
        use_cuda = True
        cpus = 0
        gpus = 1

print("Using %d cpus and/or %d gpus for step %d." % (cpus, gpus, ith_step))
# set number of pytorch threads (default=1):
# torch.set_num_threads(cpus)
torch.set_num_threads(cpu_count() // n_ensemble)
# ----


# ---- feature groups:
feature_groups_df = pd.read_excel(
    os.path.join(fileLoc, "analysis", "features.xlsx"),
    engine="openpyxl",
    sheet_name="feature_table",
).dropna(how="all")
feature_groups_df = feature_groups_df.rename(
    columns={
        "Information Source": "information",
        "Instrument Source": "instrument",
        "Group": "group",
    }
)
feature_groups_df["information"] = feature_groups_df["information"].str.lower()
feature_groups_df["instrument"] = feature_groups_df["instrument"].str.lower()

feature_groups = {}
for group in feature_groups_df["group"].unique():
    fgroup = feature_groups_df.loc[
        (feature_groups_df["group"] == group) & ~feature_groups_df["Feature"].isin(other_columns_to_delete),
        "Feature",
    ].tolist()
    if len(fgroup) > 0:
        feature_groups[group] = fgroup
        print(len(feature_groups[group]))


# ---- different sample groups:
def parser(group):
    info = group.split("_")
    features_to_keep = []
    for i in info:
        column = i.split("-")[0]
        value = i.split("-")[1].split(",")
        features_to_keep.append(feature_groups_df.loc[feature_groups_df[column].isin(value), "Feature"].tolist())
    to_keep = features_to_keep[0]
    for sublist in features_to_keep[1:]:
        to_keep = list(set(to_keep) & set(sublist))
    to_keep.sort()
    return to_keep


# set other colums to zero
if sample_type == "full":
    for sample_group in [
        "information-options",
        "information-underlying",
        "instrument-bucket;instrument-contract",
    ]:
        to_keep = []
        for group in sample_group.split(";"):
            to_keep += parser(group)
        to_kick = list(set(feature_groups_df["Feature"].tolist()) - set(to_keep))
        to_kick.sort()
        print(len(to_kick))
        feature_groups[sample_group] = list(set(to_kick))


# %%
@ray.remote(num_cpus=cpus, num_gpus=gpus)
def test_score_and_feature_importance(
    i_m: int,
    model_type: str,
    library: str,
    config_list: pd.DataFrame,
    feature_groups: dict = {},
):
    """..."""

    # --------------------
    m = m_string + "_%d" % i_m
    print("OOS fit model m=%d: %s" % (i_m, m))

    # ---- obtain prediction function depending on library used:
    if library == "sklearn":
        model = load(m)

        def predictor(X, model):
            return model.predict(X)

    elif library == "pytorch":
        config = config_list[i_m]
        model_config = config["model_params"]
        model_config["C"] = C  # input layer size
        model = init_models(model_type, model_config)
        model_state, _ = torch.load(m, map_location=torch.device("cpu"))
        model.load_state_dict(model_state)
        model.eval()  # evaluation mode, fixed batch norm and dropout layers.

        device = "cpu"
        if use_cuda:
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        model.to(device)

        def predictor(X, model):
            with torch.no_grad():
                X = torch.tensor(X).to(device)
                predicted = model(X).cpu().numpy()
            return predicted

    elif library == "lightgbm":
        model = lgbm.Booster(model_file=m)
        model.params["objective"] = "regression"

        def predictor(X, model):
            return model.predict(X, n_jobs=cpus // 2)

    else:
        raise ValueError("Wrong save_type. Not implemented yet.")

    # ---- start predictions:
    Xs = []
    ys = []
    for _, file in enumerate(files):
        X, y = read_data_single_file(
            file,
            date_column,
            target_column,
            other_columns_to_delete,
            s_test,
            e_test,
        )
        Xs.append(X)
        ys.append(y)
    Xs = pd.concat(Xs)
    ys = pd.concat(ys)

    output = pd.DataFrame()
    output["optionid"] = Xs["optionid"].values
    output["date"] = Xs.index
    Xs = Xs.drop(columns=["optionid"])
    char_names = Xs.columns
    Xs = Xs.to_numpy()
    ys = ys.to_numpy()
    output["target"] = ys
    output["predicted"] = predictor(Xs, model)

    print("Feature importance.")
    f_imp = feature_importance(Xs, model, predictor, char_names)
    output.loc[:, f_imp.keys()] = f_imp.values()
    if feature_groups:
        print("Feature importance for groups.")
        f_imp = feature_importance(Xs, model, predictor, char_names, feature_groups, group_prefix="m")
        output.loc[:, f_imp.keys()] = f_imp.values()

    print("SHAP.")
    groups_only = {
        key: values
        for key, values in feature_groups.items()
        if key
        not in [
            "information-options",
            "information-underlying",
            "instrument-bucket;instrument-contract",
        ]
    }
    if library == "lightgbm":
        SHAP_explainer = shap.TreeExplainer(model)
        SHAP_values = SHAP_explainer.shap_values(Xs)  # test inputs
    elif library == "sklearn":
        if model_type == "PCR":
            Xs_train = model.named_steps["pca"].transform(Xs)
            SHAP_explainer = shap.LinearExplainer(model["regressor"], Xs_train)
            SHAP_values = SHAP_explainer.shap_values(Xs_train)  # test inputs
            SHAP_values = model.named_steps["pca"].inverse_transform(SHAP_values)
        elif model_type in ["PLS"]:
            setattr(model, "intercept_", 0)
            setattr(model, "coef_", np.squeeze(getattr(model, "coef_")))
            SHAP_explainer = shap.LinearExplainer(model, Xs)
            SHAP_values = SHAP_explainer.shap_values(Xs)  # test inputs
            # SHAP_explainer = shap.KernelExplainer(
            #     lambda x: predictor(x, model), shap.kmeans(Xs, 10)
            # )
            # SHAP_values = SHAP_explainer.shap_values(
            #     shap.utils.sample(Xs, nsamples=1000)
            # )  # test inputs
        else:
            SHAP_explainer = shap.LinearExplainer(model, Xs)
            SHAP_values = SHAP_explainer.shap_values(Xs)  # test inputs
    else:
        SHAP_explainer = shap.DeepExplainer(model, data=torch.tensor(shap.utils.sample(Xs, nsamples=1000)).to(device))
        SHAP_values = SHAP_explainer.shap_values(torch.tensor(shap.utils.sample(Xs, nsamples=1000000)))  # test inputs
    shaps, group_shaps = SHAP_importance(SHAP_values, char_names, groups_only, "m")

    print("Returning.")
    output = output.set_index(["date", "optionid"])
    shaps.index = output.index
    group_shaps.index = output.index
    print(shaps)
    print(group_shaps.abs().mean())

    return output, shaps, group_shaps


def joblib_test_score_and_feature_importance(
    i_m: int,
    model_type: str,
    library: str,
    config_list: pd.DataFrame,
    feature_groups: dict = {},
):
    """..."""

    # --------------------
    m = m_string + "_%d" % i_m
    print("OOS fit model m=%d: %s" % (i_m, m), flush=True)

    # ---- obtain prediction function depending on library used:
    if library == "sklearn":
        model = load(m)

        def predictor(X, model):
            return model.predict(X)

    elif library == "pytorch":
        config = config_list[i_m]
        model_config = config["model_params"]
        model_config["C"] = C  # input layer size
        model = init_models(model_type, model_config)
        model_state, _ = torch.load(m, map_location=torch.device("cpu"))
        model.load_state_dict(model_state)
        model.eval()  # evaluation mode, fixed batch norm and dropout layers.

        device = "cpu"
        # if use_cuda:
        #     device = "cuda:0"
        #     if torch.cuda.device_count() > 1:
        #         model = nn.DataParallel(model)
        model.to(device)

        def predictor(X, model):
            with torch.no_grad():
                X = torch.tensor(X).to(device)
                predicted = model(X).cpu().numpy()
            return predicted

    elif library == "lightgbm":
        model = lgbm.Booster(model_file=m)
        model.params["objective"] = "regression"

        def predictor(X, model):
            return model.predict(X, n_jobs=cpus // 2)

    else:
        raise ValueError("Wrong save_type. Not implemented yet.")

    # ---- start predictions:
    Xs = []
    ys = []
    for _, file in enumerate(files):
        X, y = read_data_single_file(
            file,
            date_column,
            target_column,
            other_columns_to_delete,
            s_test,
            e_test,
        )
        Xs.append(X)
        ys.append(y)
    Xs = pd.concat(Xs)
    ys = pd.concat(ys)

    output = pd.DataFrame()
    output["optionid"] = Xs["optionid"].values
    output["date"] = Xs.index
    Xs = Xs.drop(columns=["optionid"])
    char_names = Xs.columns
    Xs = Xs.to_numpy()
    ys = ys.to_numpy()
    output["target"] = ys
    output["predicted"] = predictor(Xs, model)

    print("Feature importance.", flush=True)
    f_imp = feature_importance(Xs, model, predictor, char_names)
    output.loc[:, f_imp.keys()] = f_imp.values()
    if feature_groups:
        print("Feature importance for groups.", flush=True)
        f_imp = feature_importance(Xs, model, predictor, char_names, feature_groups, group_prefix="m")
        output.loc[:, f_imp.keys()] = f_imp.values()

    print("SHAP.", flush=True)
    groups_only = {
        key: values
        for key, values in feature_groups.items()
        if key
        not in [
            "information-options",
            "information-underlying",
            "instrument-bucket;instrument-contract",
        ]
    }
    if library == "lightgbm":
        SHAP_explainer = shap.TreeExplainer(model)
        SHAP_values = SHAP_explainer.shap_values(Xs)  # test inputs
    elif library == "sklearn":
        if model_type == "PCR":
            Xs_train = model.named_steps["pca"].transform(Xs)
            SHAP_explainer = shap.LinearExplainer(model["regressor"], Xs_train)
            SHAP_values = SHAP_explainer.shap_values(Xs_train)  # test inputs
            SHAP_values = model.named_steps["pca"].inverse_transform(SHAP_values)
        elif model_type in ["PLS"]:
            setattr(model, "intercept_", 0)
            setattr(model, "coef_", np.squeeze(getattr(model, "coef_")))
            SHAP_explainer = shap.LinearExplainer(model, Xs)
            SHAP_values = SHAP_explainer.shap_values(Xs)  # test inputs
            # SHAP_explainer = shap.KernelExplainer(
            #     lambda x: predictor(x, model), shap.kmeans(Xs, 10)
            # )
            # SHAP_values = SHAP_explainer.shap_values(
            #     shap.utils.sample(Xs, nsamples=1000)
            # )  # test inputs
        else:
            SHAP_explainer = shap.LinearExplainer(model, Xs)
            SHAP_values = SHAP_explainer.shap_values(Xs)  # test inputs
    else:
        SHAP_explainer = shap.DeepExplainer(model, data=torch.tensor(shap.utils.sample(Xs, nsamples=1_000)).to(device))
        SHAP_values = SHAP_explainer.shap_values(
            torch.tensor(shap.utils.sample(Xs, nsamples=10_000_000))
        )  # test inputs
    shaps, group_shaps = SHAP_importance(SHAP_values, char_names, groups_only, "m")

    print("Returning.", flush=True)
    output = output.set_index(["date", "optionid"])
    shaps.index = output.index
    group_shaps.index = output.index
    # print(shaps)
    # print(group_shaps.abs().mean())

    return output, shaps, group_shaps


# %%
if __name__ == "__main__":
    start = time.time()

    # ---- obtain results_table:
    print(results_table.iloc[ith_step])
    s_test, e_test = results_table[["test_start", "test_end"]].iloc[ith_step].tolist()
    files = results_table["test_files"].iloc[ith_step]

    print("Get models for step i=%d" % ith_step)
    models = glob.glob(os.path.join(model_files, model_type + "_%d_*" % ith_step))
    try:
        config_file = [m for m in models if "_configs.pkl" in m][0]
    except IndexError as e:
        print(e)
        raise IndexError("No config for step %d, skipping." % ith_step)

    # ---- obtain model files:
    models = [m for m in models if ".pkl" not in m]
    if use_refit:
        models = [m for m in models if "refit" in m]
    else:
        models = [m for m in models if "refit" not in m]
    m_string = models[0].rsplit("_", 1)[0]
    print("Working on model list:")
    for m in models:
        print("  - ", m)
    print("\n")

    # loop over models:
    best_df = pd.read_pickle(config_file)
    print(best_df)
    print(best_df.iloc[0])
    print(best_df.columns)
    config_list = best_df.loc[:, [col for col in best_df.columns if "config/" in col]]
    config_list.columns = [c.replace("config/", "") for c in config_list.columns]
    config_list = [x._asdict() for x in config_list.itertuples()]
    print(config_list)

    # ---- submission loop:
    results = Parallel(n_jobs=n_ensemble, verbose=51)(
        delayed(joblib_test_score_and_feature_importance)(
            i_m=i_m,
            model_type=model_type,
            library=library,
            config_list=config_list,
            feature_groups=feature_groups,
        )
        for i_m in range(n_ensemble)
    )

    # ---- concat results:
    predictions = results[0][0]
    shaps = results[0][1]
    group_shaps = results[0][2]
    for i_m in np.arange(1, n_ensemble):
        print(i_m)
        predictions = (predictions * i_m + results[i_m][0]) / (i_m + 1)
        shaps = (shaps * i_m + results[i_m][1]) / (i_m + 1)
        group_shaps = (group_shaps * i_m + results[i_m][2]) / (i_m + 1)

    skip_cols = ["target", "optionid", "date", "permno", "bucket"]
    if feature_importance_predictions:
        # ---- save predictions
        predictions.reset_index().to_parquet(os.path.join(model_files, "predictions_%d.pq" % ith_step))
    else:  # only return change in R2 due to feature importance, and simple predictions.
        # ---- save predictions
        predictions[["target", "predicted"]].reset_index().to_parquet(
            os.path.join(model_files, "predictions_%d.pq" % ith_step)
        )
    shaps.to_parquet(os.path.join(model_files, "shaps_%d.pq" % ith_step))
    group_shaps.to_parquet(os.path.join(model_files, "group_shaps_%d.pq" % ith_step))
    print("Total time elapsed: {:.3f} minutes.".format((time.time() - start) / 60))
