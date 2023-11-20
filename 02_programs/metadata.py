# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
import socket
import os
from pathlib import Path
import pandas as pd
import gc
import datetime
import argparse

from cross_validation_time_series import scoring_table_sample_split
from init_model_parameters import (
    init_parameters,
    init_sample,
    possible_models,
    possible_samples,
)
from data_loading import read_data_single_file

from joblib import dump, load

import torch

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# %%
host = socket.gethostname()
# ! do refit?
refit_insample = False


host = socket.gethostname()
if "D-1210" in host:
    print("Local\n")

    print("Possible samples:")
    for i, elem in enumerate(possible_samples):
        print("%d: %s" % (i, elem))
    fileLoc = input("Which file location?   ")
    fileLoc = possible_samples[int(fileLoc)]

    # ! -----------------------------------------------------
    target_column = "return_dh_daily_inv"
    # ! -----------------------------------------------------

    print("Possible models for this file:")
    for model_step, s in possible_models.items():
        print(" - %s (library %s)" % (model_step, s))
    print("\n\n")
    model_type = input("Model type   ")

    feature_groups = "information:options"

else:
    print("Remote\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model_type to use. For a list, check 'possible_models'.")
    parser.add_argument("--fileLoc", help="file location. can be in {}".format(possible_samples))
    parser.add_argument("--target_column", help="column of the values to predict.")
    parser.add_argument(
        "--skip_training",
        help="Whether to skip training. Boolean flag.",
        action="store_true",
    )
    parser.add_argument(
        "--feature_groups",
        help="Feature group to use for prediction. "
        + "Example to parse: "
        + "information-underlying_instrument-underlying;information-options_instrument-bucket"
        + "Also: 'full' to include all features.",
        default="full",
    )
    args = parser.parse_args()
    model_type = args.model
    fileLoc = args.fileLoc
    target_column = args.target_column
    feature_groups = args.feature_groups
    skip_training = args.skip_training


# ---- sample setup:
sample_setup = init_sample(fileLoc, target_column, feature_groups, rolling_estimation=False, exclude_gfc=False)  # !
date_column = sample_setup["date_column"]
other_columns_to_delete = sample_setup["other_columns_to_delete"]
initial_training_months = sample_setup["initial_training_months"]
keep_start_fixed = sample_setup["keep_start_fixed"]
validation_months = sample_setup["validation_months"]
testing_months = sample_setup["testing_months"]
model_validity_months = sample_setup["model_validity_months"]
exclude_gfc = sample_setup["exclude_gfc"]

files_to_read = sample_setup["files_to_read"]
modelLoc = fileLoc.replace("04_results", "05_models")
concatLoc = os.path.join(os.path.abspath(fileLoc), "concat")  # location for combined .pq or .bin (lightgbm) files.


# ---- model setup
model_params = init_parameters(model_type)
use_batches = model_params["hyper_params"]["use_batches"]
out_of_core = model_params["hyper_params"]["out_of_core"]
num_samples = model_params["hyper_params"]["num_samples"]

# some exception handling:
if model_type not in possible_models:
    raise ValueError("Wrong model type selected.")
library = possible_models[model_type]

if not use_batches and library == "pytorch":
    raise ValueError("Pytorch requires batching. Set 'use_batches' to True.")

if use_batches and library == "lightgbm":
    raise ValueError(
        "Lightgbm requires loading entire dataset (as binary). Set 'use_batches' to False."
        + " Also make sure that .bin files exist in 'concatLoc'."
    )


# ---- allocate resources:
if out_of_core:  # out of core calculations.
    cpus = 4
else:  # lightgbm uses smaller binary files:
    if library == "lightgbm":
        if "D-1210" in host:
            cpus = 4
        else:
            cpus = 36
    else:  # other libraries, no batching.
        if "D-1210" in host:
            cpus = 8
        else:
            cpus = 36

# ---- check for GPUs.
use_cuda = False
if library == "pytorch":
    # if gpus available, request no cpus:
    if torch.cuda.is_available():
        print("Using CUDA.")
        use_cuda = True
        cpus = 0
        if "D-1210" in host:
            gpus = 1
        else:
            gpus = 0.25
    else:
        gpus = 0
else:
    gpus = 0
# ----


# # ---- get size and relative size of current dataset:
C, _ = read_data_single_file(
    files_to_read[0],
    date_column=date_column,
    target_column=target_column,
    other_columns_to_delete=other_columns_to_delete,
)
print(C.shape)
C = C.shape[1]
# # ----


# ---- some logging for convenience:
print("Using {} gpus and/or {} cpus per trial.".format(gpus, cpus))
print("\n\n\t\t\t--- Config ---\n")
for key, value in model_params.items():
    print("{}:".format(key))
    for inner_key, inner_value in value.items():
        print("\t{}: {}".format(inner_key, inner_value))
    print("\n")
print("\n")
print("Sample setup:")
for key, value in sample_setup.items():
    if isinstance(value, list) and len(value) > 10:
        value = value[:10]
    print("{}: {}".format(key, value))
print("\n")

config_string = "Using {} gpus and {} cpus per trial.".format(gpus, cpus)
config_string += "\n\n\t\t\t--- Config ---\n"
for key, value in model_params.items():
    config_string += "{}:\n".format(key)
    for inner_key, inner_value in value.items():
        config_string += "\t{}: {}\n".format(inner_key, inner_value)
    config_string += "\n"
config_string += "\n"
config_string += "Sample setup:\n"
for key, value in sample_setup.items():
    if isinstance(value, list) and len(value) > 10:
        value = value[:10]
    config_string += "{}: {}\n".format(key, value)

# ----


# %%
progress_file = os.path.join(modelLoc, model_type + "___" + feature_groups + "_progress.txt")
steps_file = os.path.join(modelLoc, model_type + "___" + feature_groups + "_steps.txt")

# progress tracking: i.e. start at last completed trial.
# ! Only restarts if "model_type"_progress.txt is available.
# for safety purposes, always redo <fit_best_model>.
print("Progress tracking...")
if os.path.isfile(progress_file):
    print("Previous trial found. Resuming.")
    with open(progress_file, "r+") as file:
        logLoc = file.read()
    trial_datetime = logLoc.split("___")[1]
    save_str = logLoc + model_type + "_"
    in_sample_fit = []
    info_dict = load(save_str + "info_dict")
    info_dict["save_str"] = save_str
    info_dict["logLoc"] = logLoc
    dump(info_dict, save_str + "info_dict")
    results_table = pd.read_pickle(save_str + "results_table.pkl")
else:
    print("No previous trial found. Restarting.")
    # ---- make log loc
    trial_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logLoc = os.path.join(modelLoc, model_type + "___" + trial_datetime + "___" + feature_groups + "/")
    Path(logLoc + "logs").mkdir(parents=True, exist_ok=True)
    # Path(logLoc + "checkpoints").mkdir(parents=True, exist_ok=True)
    save_str = logLoc + model_type + "_"

    # log file:
    with open(save_str + "config.txt", "w") as txt:
        txt.write(config_string)

    print("Sample splits...", flush=True)
    data = pd.concat([pd.read_parquet(f, columns=["date_col"]) for f in files_to_read])
    months_in_sample, results_table = scoring_table_sample_split(
        data,
        date_column,
        initial_training_months,
        validation_months,
        testing_months,
        model_validity_months,
        keep_start_fixed,
    )
    print(months_in_sample.groupby(months_in_sample.index.year).sum())
    del data  # free up memory.

    # obtain dates in each file:
    print("Reading file bounds.", flush=True)
    dates_in_files = {}
    for f in files_to_read:
        data = pd.read_parquet(f, columns=[date_column]).squeeze()
        dates_in_files[f] = [data.min(), data.max()]
    del data
    dates_in_files = pd.DataFrame(dates_in_files, index=["start", "end"]).T
    gc.collect()

    results_table["train_files"] = None
    results_table["validation_files"] = None
    results_table["test_files"] = None
    for idx, row in results_table.iterrows():
        if exclude_gfc:
            train_files = dates_in_files[
                (dates_in_files["start"] >= row["train_start"])
                & (dates_in_files["end"] <= row["train_end"])
                & ((dates_in_files["start"] <= "2007-01-01") | (dates_in_files["end"] >= "2009-12-31"))
            ].index.tolist()
            results_table.at[idx, "train_files"] = train_files

            validation_files = dates_in_files[
                (dates_in_files["start"] >= row["validate_start"])
                & (dates_in_files["end"] <= row["validate_end"])
                & ((dates_in_files["start"] <= "2007-01-01") | (dates_in_files["end"] >= "2009-12-31"))
            ].index.tolist()
            results_table.at[idx, "validation_files"] = validation_files

            test_files = dates_in_files[
                (dates_in_files["start"] >= row["test_start"])
                & (dates_in_files["end"] <= row["test_end"])
                & ((dates_in_files["start"] <= "2007-01-01") | (dates_in_files["end"] >= "2009-12-31"))
            ].index.tolist()
            results_table.at[idx, "test_files"] = test_files

        else:
            train_files = dates_in_files[
                (dates_in_files["start"] >= row["train_start"]) & (dates_in_files["end"] <= row["train_end"])
            ].index.tolist()
            results_table.at[idx, "train_files"] = train_files

            validation_files = dates_in_files[
                (dates_in_files["start"] >= row["validate_start"]) & (dates_in_files["end"] <= row["validate_end"])
            ].index.tolist()
            results_table.at[idx, "validation_files"] = validation_files

            test_files = dates_in_files[
                (dates_in_files["start"] >= row["test_start"]) & (dates_in_files["end"] <= row["test_end"])
            ].index.tolist()
            results_table.at[idx, "test_files"] = test_files

    for item in results_table["train_files"].iloc[-1]:
        print(item)
    info_dict = {
        "feature_group": feature_groups,
        "fileLoc": fileLoc,
        "modelLoc": modelLoc,
        "concatLoc": concatLoc,
        "logLoc": logLoc,
        "date_column": date_column,
        "target_column": target_column,
        "other_columns_to_delete": other_columns_to_delete,
        "save_str": save_str,
        "library": library,
        "use_cuda": use_cuda,
        "cpus": cpus,
        "gpus": gpus,
        "C": C,
        "trial_datetime": trial_datetime,
    }

    dump(info_dict, save_str + "info_dict")
    results_table.to_pickle(save_str + "results_table.pkl")
    # --------------------------------------
print(results_table)

# create all_steps file for "model_type":
with open(steps_file, "w+") as text:
    if skip_training:
        for idx in results_table.index:
            text.write(str(idx) + "\n")
    else:
        for idx in results_table[results_table["done"] == 0].index:
            text.write(str(idx) + "\n")

# create last_trial file for "model_type":
with open(progress_file, "w+") as text:
    text.write(logLoc)


# %%
