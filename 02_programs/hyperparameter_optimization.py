# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
"""
    Code to estimate multiple machine learning models on large datasets using a
    training/validation sample split with possible subsequent post-analysis.

    Early stopping:
        - Early stopping entire experiments if consecutive "top_models"
            failed to produce sufficient variation ("std") in the scoring
            variable for at least "patience" trials.
        - Early stopping individual trials if "num_results" iteration
            failed to produce sufficient variation ("std_trial") in the scoring
            variable, or the score for "num_results" has not improved by at least
            "tol". Considers a "grace_period" which are always run before early
            stopping.

    Types of computation ----- out-of-core/batches/full sample:
        - Possible to run experiments with the entire dataset.
            Set "use_batches" to False, and use a single combined file as input.
        - Possible to run experiments with the entire dataset and batching.
            Set "use_batches" to True, and use a single combined file as input.
        - Possible to run experiments out-of-core, loading individual files
            in memory. Set "use_batches" to True, and use a list of input files.

    Restarting old trials:
        A trial can be resumed if it fails to complete (i.e. resources are
            requested by a higher priority, or a memory error occurs).
            To do so, simply rerun the program and the last experiment will be
            automatically resumed.
        To restart the entire program, delete file "model_type"_progress.csv in
            "../05_models/..".

"""

# %%
import socket
import os
from shutil import copyfile, rmtree
import glob
import pandas as pd
import numpy as np
import gc
import time
import argparse

from joblib import dump, load

from init_model_parameters import (
    init_models,
    init_parameters,
    possible_models,
    possible_samples,
)
from data_loading import read_data_parallel, DataLoader, DataIterator
from scoring import mse
from early_stopping import ExperimentAndTrialPlateauStopper
from callbacks import trial_name_creator, MyCallback

import ray
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.exceptions import RayActorError

from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import AsyncHyperBandScheduler

import torch
import torch.nn as nn

import torch.optim as optim

import lightgbm as lgbm

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# %%
def get_checkpoints_paths(logdir):
    """Finds the checkpoints within a specific folder.
    Returns a pandas DataFrame of training iterations and checkpoint
    paths within a specific folder.
    Raises:
        FileNotFoundError if the directory is not found.
    """
    marker_paths = glob.glob(os.path.join(logdir, "checkpoint_*/.is_checkpoint"))
    iter_chkpt_pairs = []
    for marker_path in marker_paths:
        chkpt_dir = os.path.dirname(marker_path)
        metadata_file = os.path.join(chkpt_dir, ".tune_metadata")
        if not os.path.isfile(metadata_file):
            print("{} has zero or more than one tune_metadata.".format(chkpt_dir))
            return None
        tmp_marker = os.path.join(chkpt_dir, ".temp_marker")
        if os.path.isfile(tmp_marker):
            print("{} marked as temporary.".format(chkpt_dir))
            return None
        chkpt_path = metadata_file[: -len(".tune_metadata")]
        chkpt_iter = int(chkpt_dir[chkpt_dir.rfind("_") + 1 :])
        iter_chkpt_pairs.append([chkpt_iter, chkpt_path])
    chkpt_df = pd.DataFrame(iter_chkpt_pairs, columns=["training_iteration", "chkpt_path"])
    return chkpt_df


# %%
host = socket.gethostname()
if "D-1210" in host:
    print("Local\n")
    LOCAL = True

    print("Possible samples:")
    for i, elem in enumerate(possible_samples):
        print("%d: %s" % (i, elem))
    fileLoc = input("Which file location?   ")
    fileLoc = possible_samples[int(fileLoc)]

    print("Possible models for this file:")
    for model_step, s in possible_models.items():
        print(" - %s (library %s)" % (model_step, s))
    print("\n\n")
    model_type = input("Model type   ")

    skip_training = False

else:
    print("Remote\n")
    LOCAL = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_nodes",
        help="number of nodes to use for this computation.",
        type=int,
        default=1,
    )
    parser.add_argument("--ith_step", help="model_type to use. For a list, check 'possible_models'.")
    parser.add_argument("--model", help="model_type to use. For a list, check 'possible_models'.")
    parser.add_argument("--fileLoc", help="file location. can be in {}".format(possible_samples))
    parser.add_argument(
        "--skip_training",
        help="Whether to skip training. Boolean flag.",
        action="store_true",
    )
    parser.add_argument(
        "--restart",
        help="Whether to skip trial resume. Boolean flag.",
        action="store_true",
    )
    parser.add_argument(
        "--feature_groups",
        help="Feature group to use for prediction. "
        + "Example to parse: "
        + "information:underlying_instrument:underlying;information:options_instrument:bucket"
        + "Also: 'full' to include all features.",
        default="full",
    )
    args = parser.parse_args()
    ith_step = int(args.ith_step)
    model_type = args.model
    fileLoc = args.fileLoc
    skip_training = args.skip_training
    feature_groups = args.feature_groups
    restart = args.restart

# get experiment location:
modelLoc = fileLoc.replace("04_results", "05_models")
progress_file = os.path.join(modelLoc, model_type + "___" + feature_groups + "_progress.txt")
steps_file = os.path.join(modelLoc, model_type + "_steps.txt")
with open(progress_file) as f:
    experimentLoc = f.readlines()[0]


# ---- read results_table and info_dict:
info_dict = load(os.path.join(experimentLoc, model_type + "_info_dict"))
concatLoc = info_dict["concatLoc"]
logLoc = info_dict["logLoc"]
date_column = info_dict["date_column"]
target_column = info_dict["target_column"]
other_columns_to_delete = info_dict["other_columns_to_delete"]
save_str = info_dict["save_str"]
library = info_dict["library"]
# use_cuda = info_dict["use_cuda"]
# gpus = info_dict["gpus"]
trial_datetime = info_dict["trial_datetime"]

results_table = pd.read_pickle(save_str + "results_table.pkl")
all_steps = results_table.index[results_table.done == 0].tolist()

# ---- model setup
model_params = init_parameters(model_type)
use_batches = model_params["hyper_params"]["use_batches"]
out_of_core = model_params["hyper_params"]["out_of_core"]
num_samples = model_params["hyper_params"]["num_samples"]
searcher = model_params["hyper_params"]["search_alg"]
num_epochs = model_params["hyper_params"]["epochs"]
if use_batches is False:
    num_epochs = 1


# ---- get number of cpus/gpus for estimation:
# NOTE: depends on size of current data set + number of features selected.
if out_of_core:
    N = results_table.loc[ith_step, "largest_train_size"]
else:
    N = results_table.loc[ith_step, "train_val_size"]
C = info_dict["C"]
mem_required = (N * C) * 4  # bytes needed
mem_required /= 1_000_000_000  # GB needed
mem_required *= 2
if library == "lightgbm":  # smaller binary files.
    mem_required /= 4

cpus = int(max(1, 36 / (90 / mem_required)))  # some safeguarding against excessively large datasets (max cores = 72).
possible_cpus = np.array([2, 4, 6, 8, 9, 12, 18, 24, 36, 72, 100000])
cpus = int(possible_cpus[np.where(cpus <= possible_cpus)[0][0]])
cpus = min(36, cpus)
if library == "lightgbm":
    cpus = max(2, cpus)

cpu_num = 32
if library == "lightgbm":
    cpus = cpu_num // 16
    # cpus = cpu_num // 8  # for weekly return horizon
else:
    cpus = cpu_num // 32

# GPUS:
use_cuda = False
if library == "pytorch":
    # if gpus available, request no cpus:
    if torch.cuda.is_available():
        print("Using CUDA.")
        print(torch.cuda.get_device_capability())
        use_cuda = True
        cpus = 0
        gpus = 1
    else:
        gpus = 0
else:
    gpus = 0

print("Using %d cpus and/or %f gpus for step %d." % (cpus, gpus, ith_step))

# set number of pytorch threads (default=1):
torch.set_num_threads(max(1, cpus // 2))


# ------------- stopping mechanism -----------------------------------------
# ---- trial-level stopping parameters also used in refit_ensemble.
trial_grace_period = model_params["tune_params"]["trial_grace_period"]
trial_num_results = model_params["tune_params"]["trial_num_results"]
trial_tol = model_params["tune_params"]["trial_tol"]
# ---- experiment-level stopping parameters also used in refit_ensemble.
exp_top_models = model_params["tune_params"]["exp_top_models"]
# --------------------------------------------------------------------------


# %%
@ray.remote(num_cpus=cpus)  # only cpus! ray tune head node.
def model_fit_parallel_optimization(
    ith_step: int,
    results_table: pd.DataFrame,
    model_type: str,
    use_batches: bool = False,
    logLoc: str = "./",
    trial_name: str = "trial",
    C: int = None,
):
    """..."""

    def train_model(
        config,
        checkpoint_dir=None,
        model_type=None,
        s_train=None,
        e_train=None,
        s_val=None,
        e_val=None,
        X_training=None,
        y_training=None,
        X_validation=None,
        y_validation=None,
    ):
        warnings.filterwarnings("ignore", category=UserWarning)  # CUDA warnings.
        if not model_type:
            raise ValueError("Expected model input, currently = None.")

        # set up hyperparameters
        model_config = config["model_params"]
        optim_config = config["optim_params"]
        sched_config = config["sched_params"]
        swa_config = config["swa_params"]
        hyper_config = config["hyper_params"]

        if "batch_size" in hyper_config.keys():
            batch_size = hyper_config["batch_size"]
        else:
            batch_size = 0
        if "epochs" in hyper_config.keys():
            epochs = hyper_config["epochs"]
        else:
            epochs = 0

        if library == "pytorch":
            # model selection
            model_config["C"] = C  # input layer size
            model = init_models(model_type, model_config)

            # available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
            # print(available_gpus, flush=True)
            # NOTE: ray workers only have access to their respective GPU.
            device = "cpu"
            # model = DistributedDataParallel(model, find_unused_parameters=True)
            if use_cuda:
                device = "cuda:0"
                # if torch.cuda.device_count() > 1:
                #     model = nn.DataParallel(model)
            model.to(device)

            criterion = nn.MSELoss()
            # optimizer = optim.SGD(
            #     model.parameters(),
            #     **optim_config,
            # )
            optimizer = optim.AdamW(model.parameters(), **optim_config)
            sched = sched_config["sched"]
            del sched_config["sched"]
            if sched == "Cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_config)
            elif sched == "SGDR":
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **sched_config)
            elif sched == "OneCycle":
                # TODO: give size of samples to this function to use with OneCycleLR
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    **sched_config,
                )
            else:
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)  # keep lr constant.

            # ---- swa scheduling:
            if swa_config:
                swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, **swa_config)
                swa_model = torch.optim.swa_utils.AveragedModel(model)
            else:
                swa_scheduler = None
                swa_model = None

            swa_started = False
            swa_epochs = epochs
            train_scores = []
            scores = []
            estimation_times = dict(training=0, validation=0)
            for epoch in range(epochs):

                t = time.time()
                # training
                train_score = 0.0
                train_steps = 0
                if out_of_core:
                    for X, y in DataLoader(
                        files_to_read=files_train,
                        out_of_core=True,
                        date_column=date_column,
                        target_column=target_column,
                        other_columns_to_delete=other_columns_to_delete,
                        start=s_train,
                        end=e_train,
                        batch_size=batch_size,
                        shuffle_data=True,
                    ):
                        X_len = len(X)
                        X, y = torch.tensor(X).to(device), torch.tensor(y.reshape(-1, 1)).to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = model(X)
                        loss = criterion(outputs, y)
                        loss.backward()
                        optimizer.step()

                        train_score += loss.item() * X_len
                        train_steps += X_len
                else:
                    for X, y in DataIterator(X_training, y_training, batch_size=batch_size, shuffle_data=True):
                        X_len = len(X)
                        X, y = torch.tensor(X).to(device), torch.tensor(y.reshape(-1, 1)).to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad(set_to_none=True)

                        # forward + backward + optimize
                        outputs = model(X)
                        loss = criterion(outputs, y)
                        loss.backward()
                        optimizer.step()

                        train_score += loss.item() * X_len
                        train_steps += X_len
                print("Training took %.5f minutes." % ((time.time() - t) / 60))
                estimation_times["training"] += time.time() - t

                del X, y
                gc.collect()

                t = time.time()
                # validation
                val_score = 0.0
                val_steps = 0
                if out_of_core:
                    for X, y in DataLoader(
                        files_to_read=files_val,
                        out_of_core=True,
                        date_column=date_column,
                        target_column=target_column,
                        other_columns_to_delete=other_columns_to_delete,
                        start=s_val,
                        end=e_val,
                        batch_size=batch_size,
                        shuffle_data=False,
                    ):
                        with torch.no_grad():
                            X_len = len(X)
                            X, y = torch.tensor(X).to(device), torch.tensor(y.reshape(-1, 1)).to(device)
                            outputs = model(X)
                            loss = criterion(outputs, y)
                            val_score += loss.item() * X_len
                            val_steps += X_len
                else:
                    for X, y in DataIterator(X_validation, y_validation, batch_size=-1, shuffle_data=False):
                        with torch.no_grad():
                            X, y = torch.tensor(X).to(device), torch.tensor(y.reshape(-1, 1)).to(device)
                            outputs = model(X)
                            loss = criterion(outputs, y)
                            val_score = loss.item()  # only one iteration, no need to calculate 'X_len'
                            val_steps = 1
                print("Validation took %.5f minutes." % ((time.time() - t) / 60))
                estimation_times["validation"] += time.time() - t
                del X, y
                gc.collect()

                for key in estimation_times:
                    print("%s: %.5f minutes" % (key, estimation_times[key] / 60))
                print(optimizer.param_groups[0]["lr"])

                # save train/val scores and report.
                train_scores.append(train_score / train_steps)
                scores.append(val_score / val_steps)
                print("Epoch {} - train_score: {}; val_score: {}".format(epoch, train_scores, scores))

                # get new learning rate:
                if swa_config:
                    # early stopping:
                    best = min(scores)
                    no_improvement = 0
                    ii = np.argmin(scores)
                    for i in range(ii, len(scores)):
                        if (scores[i] + trial_tol) >= best:
                            no_improvement += 1
                    no_improvement -= 1
                    print(
                        "Epoch: %d; No improv trials: %d; num_results to early stop: %d; swa_started? %s; grace_period %d"
                        % (
                            epoch,
                            no_improvement,
                            trial_num_results,
                            swa_started,
                            trial_grace_period,
                        )
                    )
                    if no_improvement >= trial_num_results and swa_started is False and epoch >= trial_grace_period:
                        print("SWA:::Would stop early, but in SWA context at step i=%d, epoch=%d" % (ith_step, epoch))
                        swa_started = True
                        swa_epochs = (
                            epoch + hyper_config["swa_epochs"]
                        )  # early stopping --> still same amount of averaging epochs!
                        print("Swa until epoch: %d" % swa_epochs)
                    if epoch >= hyper_config["swa_start"] or swa_started:
                        if swa_started is False:
                            print(
                                "SWA:::Starting stochastic weight averaging at step i=%d, epoch=%d" % (ith_step, epoch)
                            )
                            swa_started = True
                        swa_model.update_parameters(model)
                        swa_scheduler.step()

                    # for last epoch do a forward pass to update batch norm statistics.
                    # see https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging.
                    if epoch == (swa_epochs - 1):
                        print("Doing forward pass over data to update batch norm for SWA.")
                        if out_of_core:
                            for X, y in DataLoader(
                                files_to_read=files_train,
                                out_of_core=True,
                                date_column=date_column,
                                target_column=target_column,
                                other_columns_to_delete=other_columns_to_delete,
                                start=s_train,
                                end=e_train,
                                batch_size=batch_size,
                                shuffle_data=True,
                            ):
                                X, y = (
                                    torch.tensor(X).to(device),
                                    torch.tensor(y.reshape(-1, 1)).to(device),
                                )
                                # forward pass
                                outputs = model(X)
                        else:
                            for X, y in DataIterator(X_training, y_training, batch_size=-1, shuffle_data=True):
                                X, y = (
                                    torch.tensor(X).to(device),
                                    torch.tensor(y.reshape(-1, 1)).to(device),
                                )
                                # forward pass
                                outputs = model(X)
                        del X, y
                        gc.collect()
                        break  # stop execution after swa_epochs ran through.
                    else:
                        scheduler.step()
                else:
                    scheduler.step()

                # Here we save a checkpoint. It is automatically registered with
                # Ray Tune and will potentially be passed as the `checkpoint_dir`
                # parameter in future iterations.
                # with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
                #     path = os.path.join(checkpoint_dir, "checkpoint")
                #     torch.save((model.state_dict(), optimizer.state_dict()), path)
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)

                tune.report(
                    score=(val_score / val_steps),
                    train_score=(train_score / train_steps),
                    epoch=epoch,
                )
                gc.collect()

        elif library == "sklearn":
            model = init_models(model_type, model_config)

            if use_batches:  # incremental training
                for epoch in range(epochs):
                    # training
                    train_score = 0.0
                    train_steps = 0
                    if out_of_core:
                        for X, y in DataLoader(
                            files_to_read=files_train,
                            out_of_core=True,
                            date_column=date_column,
                            target_column=target_column,
                            other_columns_to_delete=other_columns_to_delete,
                            start=s_train,
                            end=e_train,
                            batch_size=batch_size,
                            shuffle_data=True,
                        ):
                            model.partial_fit(X, y)
                            score = mse(y, model.predict(X))
                            train_score += score * X.shape[0]
                            train_steps += X.shape[0]
                    else:
                        for X, y in DataIterator(X_training, y_training, batch_size=batch_size, shuffle_data=True):
                            model.partial_fit(X, y)
                            score = mse(y, model.predict(X))
                            train_score += score * X.shape[0]
                            train_steps += X.shape[0]
                    del X, y
                    gc.collect()

                    # validation
                    val_score = 0.0
                    val_steps = 0
                    if out_of_core:
                        for X, y in DataLoader(
                            files_to_read=files_val,
                            out_of_core=True,
                            date_column=date_column,
                            target_column=target_column,
                            other_columns_to_delete=other_columns_to_delete,
                            start=s_val,
                            end=e_val,
                            batch_size=batch_size,
                            shuffle_data=True,
                        ):
                            score = mse(y, model.predict(X))
                            val_score += score * X.shape[0]
                            val_steps += X.shape[0]
                    else:
                        for X, y in DataIterator(X_validation, y_validation, batch_size=-1, shuffle_data=True):
                            score = mse(y, model.predict(X))
                            val_score += score * X.shape[0]
                            val_steps = 1

                    # Here we save a checkpoint. It is automatically registered with
                    # Ray Tune and will potentially be passed as the `checkpoint_dir`
                    # parameter in future iterations.
                    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        dump(model, path)

                    tune.report(
                        score=(val_score / val_steps),
                        train_score=(train_score / train_steps),
                        epoch=epoch,
                    )
                    del X, y
                    gc.collect()

            else:  # regular full-sample training
                # fit model
                model.fit(X_training, y_training)
                train_score = mse(y_training, model.predict(X_training))

                # validation scoring
                score = mse(y_validation, model.predict(X_validation))

                # Here we save a checkpoint. It is automatically registered with
                # Ray Tune and will potentially be passed as the `checkpoint_dir`
                # parameter in future iterations.
                with tune.checkpoint_dir(step=0) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    dump(model, path)

                if model_type in ["PLS", "PCR"]:
                    epoch = 1
                else:
                    epoch = model.n_iter_
                tune.report(score=score, train_score=train_score, epoch=epoch)
                gc.collect()
            gc.collect()

        elif library == "lightgbm":
            model_config["num_threads"] = cpus // 2

            if use_batches:
                first_iter = True
                os.makedirs(os.path.join(logLoc, "lightgbm"), exist_ok=True)
                model_str = os.path.join(logLoc, "lightgbm", "model_%d.txt" % ith_step)
                for epoch in range(epochs):
                    # training
                    train_score = 0.0
                    train_steps = 0
                    for X, y in DataLoader(
                        files_to_read=files_train,
                        out_of_core=True,
                        date_column=date_column,
                        target_column=target_column,
                        other_columns_to_delete=other_columns_to_delete,
                        start=s_train,
                        end=e_train,
                        batch_size=batch_size,
                        shuffle_data=True,
                    ):
                        dataset = lgbm.Dataset(X, label=y)
                        if first_iter:
                            model = lgbm.train(model_config, train_set=dataset, num_boost_round=100, **optim_config)
                            model.save_model(model_str, num_iteration=model.best_iteration)
                            first_iter = False
                        else:
                            model = lgbm.train(
                                model_config,
                                train_set=dataset,
                                num_boost_round=100,
                                **optim_config,
                                init_model=model_str,
                                keep_training_booster=True,
                                num_threads=cpus // 2,
                            )
                            model.save_model(model_str, num_iteration=model.best_iteration)
                        score = mse(y, model.predict(X))
                        train_score += score * X.shape[0]
                        train_steps += X.shape[0]

                    # validation
                    val_score = 0.0
                    val_steps = 0
                    for X, y in DataLoader(
                        files_to_read=files_val,
                        out_of_core=True,
                        date_column=date_column,
                        target_column=target_column,
                        other_columns_to_delete=other_columns_to_delete,
                        start=s_val,
                        end=e_val,
                        batch_size=batch_size,
                        shuffle_data=True,
                    ):
                        score = mse(y, model.predict(X))
                        val_score += score * X.shape[0]
                        val_steps += X.shape[0]

                    # Here we save a checkpoint. It is automatically registered with
                    # Ray Tune and will potentially be passed as the `checkpoint_dir`
                    # parameter in future iterations.
                    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        dump(model, path)

                    tune.report(
                        score=(val_score / val_steps),
                        train_score=(train_score / train_steps),
                        epoch=epoch,
                    )
                    del X, y
                    gc.collect()

            else:
                train_data = lgbm.Dataset(os.path.join(concatLoc, "%d_train_%s.bin" % (ith_step, feature_groups)))
                val_data = lgbm.Dataset(os.path.join(concatLoc, "%d_val_%s.bin" % (ith_step, feature_groups)))

                evals_result = {}
                model = lgbm.train(
                    model_config,
                    evals_result=evals_result,
                    valid_sets=[val_data, train_data],
                    valid_names=["validation", "training"],
                    train_set=train_data,
                    # valid_sets=val_data,
                    learning_rates=lambda iter: model_config["learning_rate"] / ((iter + 1) ** 0.1),
                    **optim_config,
                )
                model.free_dataset()
                del train_data, val_data
                gc.collect()

                with tune.checkpoint_dir(step=0) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    model.save_model(path)
                epoch = model.current_iteration()
                train_score = model.best_score["training"]["l2"]
                score = model.best_score["validation"]["l2"]

                tune.report(score=score, train_score=train_score, epoch=epoch)

        else:
            raise ValueError("Wrong library selected.")
        # ------------------

    start = time.time()

    print("Obtain training and validation files for ith_step=%d." % ith_step)
    # sample splits
    s_train, e_train = results_table[["train_start", "train_end"]].iloc[ith_step].tolist()
    s_val, e_val = results_table[["validate_start", "validate_end"]].iloc[ith_step].tolist()
    files_train = results_table["train_files"].iloc[ith_step]
    files_val = results_table["validation_files"].iloc[ith_step]

    if (out_of_core is False) & (library != "lightgbm"):
        # ----- read in data:
        X_training, y_training = read_data_parallel(
            files_to_read=files_train,
            date_column=date_column,
            target_column=target_column,
            other_columns_to_delete=other_columns_to_delete,
            start=s_train,
            end=e_train,
            n_jobs=cpus,
        )
        X_validation, y_validation = read_data_parallel(
            files_to_read=files_val,
            date_column=date_column,
            target_column=target_column,
            other_columns_to_delete=other_columns_to_delete,
            start=s_val,
            end=e_val,
            n_jobs=cpus,
        )
        X_training = X_training.to_numpy()
        y_training = y_training.to_numpy()
        X_validation = X_validation.to_numpy()
        y_validation = y_validation.to_numpy()
        # -----
    else:
        X_training = None
        y_training = None
        X_validation = None
        y_validation = None

    print("Hyperparameter optimization for ith_step=%d." % ith_step)
    config = init_parameters(model_type)
    t_config = config["tune_params"]
    h_config = config["hyper_params"]

    if not use_batches:
        h_config["epochs"] = 0
        h_config["batch_size"] = 0

    stopper = ExperimentAndTrialPlateauStopper(
        metric="score",
        mode="min",
        epochs=num_epochs,
        trial_logfile=logLoc + "logs/%s_%d_trial_log.txt" % (model_type, ith_step),
        exp_logfile=logLoc + "logs/%s_%d_early_stopping_log.txt" % (model_type, ith_step),
        **t_config,
    )

    if searcher == "HEBO":
        max_concurrent = 16
        search_alg = HEBOSearch(metric="score", mode="min", max_concurrent=max_concurrent)
    elif searcher == "123":
        search_alg = "TEST"
    else:  # random + grid search
        search_alg = BasicVariantGenerator()

    # ! NOTE: cannot use ASHA with experiment-level early stopping,
    # ! as ASHA-stopped trials will not show up in early_stopping loop!
    scheduler = AsyncHyperBandScheduler(
        grace_period=trial_num_results if use_batches else 100,
        max_t=h_config["epochs"] if h_config["epochs"] > 0 else 100,
        reduction_factor=4,
    )

    # ---- check whether to resume
    # NOTE: tune.with_parameters automatically puts objects in object store.
    # performs a random search. Otherwise the "search_alg" can be specified.
    local_dir = "/home/h/h_beck18/ray_results/"
    tune_params = dict(
        local_dir=local_dir,
        scheduler=scheduler,
        search_alg=search_alg,
        stop=stopper,
        callbacks=[
            MyCallback(
                "min",
                logLoc + "logs/%s_%d_experiment_log.txt" % (model_type, ith_step),
                t_config["exp_top_models"],
                t_config["exp_num_results"],
                t_config["exp_grace_period"],
                t_config["exp_tol"],
            )
        ],
        resources_per_trial={"gpu": gpus, "cpu": cpus},
        config=config,
        num_samples=num_samples,
        verbose=1,
        metric="score",
        mode="min",
        reuse_actors=False,  # !
        max_failures=5,
        # fail_fast=True,
        checkpoint_score_attr="min-score",
        # sync_on_checkpoint=False,
        name=trial_name + "_%d" % ith_step,
        trial_dirname_creator=trial_name_creator,
    )

    def checkpoint_exists(directory):
        if not os.path.exists(directory):
            return False
        return any(
            (fname.startswith("experiment_state") and fname.endswith(".json")) for fname in os.listdir(directory)
        )

    resume = False
    if restart:
        try:
            rmtree(os.path.join(local_dir, trial_name + "_%d" % ith_step))
        except FileNotFoundError:
            pass
    else:
        if results_table.iloc[ith_step]["trials_exist"] == 1:
            print(
                "Does checkpoint exist?",
                checkpoint_exists(os.path.join(local_dir, trial_name + "_%d" % ith_step)),
            )
            if checkpoint_exists(os.path.join(local_dir, trial_name + "_%d" % ith_step)):
                resume = True  # True

    results_table = pd.read_pickle(save_str + "results_table.pkl")
    results_table.loc[ith_step, "trials_exist"] = 1
    results_table.to_pickle(save_str + "results_table.pkl")
    if skip_training is False:
        try:
            if resume:
                print("Trying to resume.")
            analysis = tune.run(
                tune.with_parameters(
                    train_model,
                    checkpoint_dir=logLoc,
                    model_type=model_type,
                    s_train=s_train,
                    e_train=e_train,
                    s_val=s_val,
                    e_val=e_val,
                    X_training=X_training,
                    y_training=y_training,
                    X_validation=X_validation,
                    y_validation=y_validation,
                ),
                resume=resume,
                **tune_params,
            )

        except (RayActorError, ValueError, KeyError):
            print("Resume failed. Restarting trial.")
            time.sleep(10)
            analysis = tune.run(
                tune.with_parameters(
                    train_model,
                    checkpoint_dir=logLoc,
                    model_type=model_type,
                    s_train=s_train,
                    e_train=e_train,
                    s_val=s_val,
                    e_val=e_val,
                ),
                resume=False,
                **tune_params,
            )

        estimation_time = (time.time() - start) / 60
        print("Best hyperparameters found were: ", analysis.best_config)
        print("Estimation took {:.3f} minutes.".format(estimation_time))

    # ---- save best config dataframe:
    analysis = ExperimentAnalysis(local_dir + trial_name + "_%d" % ith_step)
    dfs = analysis.trial_dataframes
    cfs = analysis.get_all_configs()
    best_df = []
    for trial in dfs.keys():
        if len(dfs[trial]) > 0:
            tmp = dfs[trial]
            if "score" not in tmp.columns:  # faulty recording, skip.
                continue
            tmp["path"] = trial
            cf = pd.Series(cfs[trial]).to_frame().T
            cf.columns = ["config/" + str(c) for c in cf.columns]
            tmp = pd.concat((tmp, cf), axis=1)
            for col in tmp.columns:
                if "config/" in col:
                    tmp[col] = tmp[col].ffill()
            best_df.append(tmp)

    best_df = pd.concat(best_df).reset_index(drop=True)
    best_df = best_df.sort_values("score")
    best_df.to_pickle(save_str + "%d_all_trials.pkl" % ith_step)

    # ---- retrieve up to "n_models" best checkpoints save in logLoc.
    n_models = 15
    if model_type == "FFNN_SWA":
        print("SWA ::: using last score, instead of best.")
        best_models = best_df.sort_values("training_iteration").groupby("trial_id").last().sort_values("score")
    else:
        best_models = best_df.groupby("trial_id").first().sort_values("score")

    n_models_transfered = 0
    for i, (idx, row) in enumerate(best_models.iterrows()):  # copy n models over
        print("Copying best trials in step i=%d (%d-best model, path=%s)" % (ith_step, i, idx))
        print(row)
        print(row["path"])
        best_iteration = row["training_iteration"] - 1
        paths = get_checkpoints_paths(row["path"])
        print(paths)
        print("Looking for best iteration %d." % best_iteration)
        if isinstance(paths, pd.DataFrame):
            print(paths.chkpt_path.iloc[0])
            path = paths.loc[paths["training_iteration"] == best_iteration, "chkpt_path"].iloc[0]
            copyfile(
                path + "checkpoint",
                logLoc + "%s_%d_%d" % (model_type, ith_step, n_models_transfered),
            )
            n_models_transfered += 1
            if n_models_transfered == n_models:  # n_models models transfered.
                break
        else:
            best_models.loc[idx] = np.nan  # gets rid of fauly trials + temporary trials.
    best_models.dropna(how="all").iloc[:n_models].to_pickle(save_str + "%d_configs.pkl" % ith_step)

    # ---- save results_df:
    tmp_results = pd.read_pickle(save_str + "results_table.pkl")
    tmp_results.loc[ith_step, "done"] = 1
    tmp_results.to_pickle(save_str + "results_table.pkl")

    return analysis._experiment_dir


# %%
if __name__ == "__main__":
    if (results_table.iloc[ith_step]["done"] == 1) & (skip_training is False) & (restart is False):
        print("Step %d already done. Skipping." % ith_step)
    else:
        start = time.time()
        print("Starting cluster...", flush=True)
        if LOCAL:
            ray.init(include_dashboard=False)
        else:
            print("Remote cluster...", flush=True)
            os.makedirs("/tmp/h_beck18", exist_ok=True)
            # ray.init(address=os.environ["ip_head"], _redis_max_memory=1024 ** 3)
            ray.init(_temp_dir="/tmp/h_beck18", address=os.environ["ip_head"], _redis_max_memory=1024**3)
        print("Cluster started.", flush=True)

        # ---- start anew, trial not done:
        if restart:
            results_table = pd.read_pickle(save_str + "results_table.pkl")
            results_table.loc[ith_step, "done"] = 0
            results_table.to_pickle(save_str + "results_table.pkl")

        # ---- run experiments ----
        experiment_dir = []
        print("Adding ith_step=%d." % ith_step)
        print(results_table.iloc[ith_step])
        experiment_dir.append(
            model_fit_parallel_optimization.remote(
                ith_step=ith_step,
                results_table=results_table,
                model_type=model_type,
                use_batches=use_batches,
                logLoc=logLoc,
                trial_name=model_type + "___" + trial_datetime,
                C=C,
            )
        )
        experiment_dir = ray.get(experiment_dir)
        results_table = pd.read_pickle(save_str + "results_table.pkl")
        results_table[["done", "trials_exist"]].to_csv(save_str + "_results_table.csv")

        # if (results_table["done"] == 1).all():
        #     os.remove(progress_file)  # trial completed with all steps.
        #     os.remove(steps_file)  # trial completed with all steps.
        estimation_time = (time.time() - start) / 60
        print("Estimation took {:.3f} minutes.".format(estimation_time))
        print("Total time elapsed: {:.3f} minutes.".format((time.time() - start) / 60))


# %%
