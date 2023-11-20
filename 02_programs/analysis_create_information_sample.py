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
from joblib import dump

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

samples = list(set([model_type.split("___")[-1] for model_type in model_types]))

all_samples = {}
for sample in samples:
    all_models = {}
    for model_type in model_types:
        if model_type.split("___")[-1] == sample:
            all_models[model_type.split("___")[0] + "_" + model_type.split("___")[-1]] = os.path.join(
                modelLoc, model_type
            )
    all_samples[sample] = all_models

print("\n\n--- Models chosen ---\n")
for m in all_models:
    print(m)
    print(all_models[m])
    print("\n")
print("\n")

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


def create_ensemble_predictions(N_files, models, all_models, model_name: str):
    print(model_name, flush=True)
    print(all_models)
    class_groups = pd.read_parquet(os.path.join(analysisLoc, "class_groups.pq"))
    class_groups = class_groups.drop(columns=[c for c in class_groups.columns if "return_" in c or "gain_" in c])
    class_columns = class_groups.columns.tolist()
    class_columns = [c for c in class_columns if c not in ["date", "optionid", "permno"]]
    skip_cols = ["target", "optionid", "date", "port", "permno"] + class_columns

    predictions = []
    errors = defaultdict(list)
    xs_errors = defaultdict(list)
    SHAPs = defaultdict(list)
    rel_SHAPs = defaultdict(list)
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
            pred["date"] = pred["date"].dt.to_period("M")
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
            shaps["date"] = shaps["date"].dt.to_period("M")
            shaps = shaps.set_index(["date", "optionid"])
            shaps = shaps.sort_index()  # make sure that every value is sorted
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
    MEM = int(30 * 1024)
    CORES = 16
    cluster = SLURMCluster(
        queue="normal,largesmp,bigsmp,requeue",
        cores=1,
        processes=1,
        memory=str(MEM) + "MB",
        walltime="04:00:00",
        local_directory="/scratch/tmp/h_beck18/",
        extra=["--resources memory=1"],
        job_extra=[
            "-J options",
            "-c " + str(CORES),
            "-o " + logFile,
            "-e " + logFile.replace(".out", ".err"),
        ],
    )
    cluster.adapt(minimum=1, maximum=len(model_types), target_duration="1s")

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

    N_files = len(glob.glob(os.path.join(list(all_models.values())[0], "predictions*")))
    df_futures = [
        client.submit(
            create_ensemble_predictions,
            N_files=N_files,
            models=linear_models,
            all_models=all_samples[sample_type],
            model_name="L-En_%s" % sample_type,
            resources={"memory": 1},
        )
        for sample_type in all_samples
    ]  # linear ensembles.
    for sample_type in all_samples:
        df_futures.append(
            client.submit(
                create_ensemble_predictions,
                N_files=N_files,
                models=nonlinear_models,
                all_models=all_samples[sample_type],
                model_name="N-En_%s" % sample_type,
                resources={"memory": 1},
            )
        )  # nonlinear ensembles.

    prediction_dict = {}
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

    dump(prediction_dict, os.path.join(saveLoc, "prediction_dict_samples.pkl"))

    print("Done.")


# %%
