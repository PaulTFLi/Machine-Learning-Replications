# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
from typing import Union
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor

from ray import tune

import torch.nn as nn
import torch.nn.functional as F

import lightgbm as lgbm

import socket
import os
import pandas as pd
import re
import glob

host = socket.gethostname()


# %% Models and Samples
possible_models = {
    "ElasticNet": "sklearn",
    "Lasso": "sklearn",
    "Ridge": "sklearn",
    "PCR": "sklearn",
    "PLS": "sklearn",
    "FFNN": "pytorch",
    "FFNN_test": "pytorch",
    "FFNN_SGDR": "pytorch",
    "FFNN_SWA": "pytorch",
    "LightGBR": "lightgbm",
    "LightDart": "lightgbm",
    "LightRF": "lightgbm",
}

possible_samples = [
    "../04_results/stock_sample/",
    "../04_results/option_sample/",
    "../04_results/option_stock_sample/",
]


# %% Model Classes
class FFNN(nn.Module):
    def __init__(self, C, l1, num_hidden_layers, p):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(C, l1)
        self.bn1 = nn.BatchNorm1d(num_features=l1)
        self.fc2 = nn.Linear(l1, l1 // 2)
        self.bn2 = nn.BatchNorm1d(num_features=l1 // 2)
        self.fc3 = nn.Linear(l1 // 2, l1 // 4)
        self.bn3 = nn.BatchNorm1d(num_features=l1 // 4)
        self.fc4 = nn.Linear(l1 // 4, l1 // 8)
        self.bn4 = nn.BatchNorm1d(num_features=l1 // 8)
        self.fc5 = nn.Linear(l1 // 8, l1 // 16)
        self.bn5 = nn.BatchNorm1d(num_features=l1 // 16)
        self.fcout = nn.Linear(l1 // (2 ** (num_hidden_layers - 1)), 1)
        self.dropout = nn.Dropout(p=p)
        self.num_hidden_layers = num_hidden_layers

    def forward(self, x):
        x = F.gelu(self.dropout(self.bn1(self.fc1(x))))
        if self.num_hidden_layers > 1:
            x = F.gelu(self.dropout(self.bn2(self.fc2(x))))
        if self.num_hidden_layers > 2:
            x = F.gelu(self.dropout(self.bn3(self.fc3(x))))
        if self.num_hidden_layers > 3:
            x = F.gelu(self.dropout(self.bn4(self.fc4(x))))
        if self.num_hidden_layers > 4:
            x = F.gelu(self.dropout(self.bn5(self.fc5(x))))
        x = self.fcout(x)
        return x


# %% Initialize sample:
def init_sample(
    fileLoc,
    target_column,
    feature_groups: Union[str, list] = "full",
    rolling_estimation: bool = False,
    exclude_gfc: bool = False,
):
    """Initialize sample splits and column setup.

    Example for feature_groups:
        feature_groups = (
            "information-underlying_instrument-underlying;information-options_instrument-bucket"
        )

    """

    if "stock_sample" in fileLoc:
        date_column = "date_col"
        other_columns_to_delete = ["permno", "small", "value", "low_beta"]

        if "D-1210" in host:
            initial_training_months = 6
            validation_months = 6
            testing_months = 6
            model_validity_months = 6
            keep_start_fixed = False
        else:
            initial_training_months = 120
            validation_months = 60
            testing_months = 12
            model_validity_months = 12
            keep_start_fixed = False

        files_to_read = [os.path.abspath(f) for f in glob.glob(fileLoc + "/*.pq") if re.match(r".*([1-2][0-9]{3})", f)]

    elif "option_sample" in fileLoc:
        date_column = "date_col"
        print("Choose target_column:")
        possible_targets = [
            "return_option",
            "gain_dh_daily",
            # "gain_dh_initial",
            "return_dh_daily_s",
            # "return_dh_initial_s",
            "return_dh_daily_o",
            # "return_dh_initial_o",
            "return_dh_daily_inv",
            # "return_dh_initial_inv",
            "return_dh_daily_margin",
            "margin_denominator",
            "margin_denominator_signed",
            "return_delevered",
            "leverage",
            "S_spreads_paid",
        ]
        for i, elem in enumerate(possible_targets):
            print("%d: %s" % (i, elem))
        print("\n")
        possible_targets.remove(target_column)

        # ---- read feature table:
        feature_table = pd.read_excel(
            os.path.join(fileLoc, "analysis", "features.xlsx"),
            engine="openpyxl",
            sheet_name="feature_table",
            usecols=["Feature", "Information Source", "Instrument Source", "Group"],
        ).dropna(how="all")
        feature_table = feature_table.rename(
            columns={
                "Information Source": "information",
                "Instrument Source": "instrument",
                "Group": "group",
            }
        )
        feature_table["information"] = feature_table["information"].str.lower()
        feature_table["instrument"] = feature_table["instrument"].str.lower()

        def parser(group):
            info = group.split("_")
            features_to_keep = []
            for i in info:
                column = i.split("-")[0]
                value = i.split("-")[1].split(",")
                features_to_keep.append(feature_table.loc[feature_table[column].isin(value), "Feature"].tolist())
            to_keep = features_to_keep[0]
            for sublist in features_to_keep[1:]:
                to_keep = list(set(to_keep) & set(sublist))
            to_keep.sort()
            return to_keep

        if feature_groups == "full":
            other_columns_to_delete = ["permno", "optionid", "bucket"] + possible_targets
        else:
            to_keep = []
            for group in feature_groups.split(";"):
                to_keep += parser(group)
            to_kick = list(set(feature_table["Feature"].tolist()) - set(to_keep))
            to_kick.sort()
            other_columns_to_delete = ["permno", "optionid", "bucket"] + to_kick + possible_targets

        if "D-1210" in host:
            initial_training_months = 6
            validation_months = 6
            testing_months = 6
            model_validity_months = 6
            keep_start_fixed = False
        else:
            initial_training_months = 60
            validation_months = 24
            testing_months = 12
            model_validity_months = 12
            keep_start_fixed = True
            if rolling_estimation:
                initial_training_months = 120
                keep_start_fixed = False

            if exclude_gfc:
                initial_training_months = 168

        files_to_read = [
            os.path.abspath(f)
            for f in glob.glob(fileLoc + "/characteristics_*.pq")
            if re.match(r".*([1-2][0-9]{3})", f)
        ]
        files_to_read.sort()

    else:
        raise ValueError("init_sample:::Wrong file location.")

    return dict(
        date_column=date_column,
        other_columns_to_delete=other_columns_to_delete,
        initial_training_months=initial_training_months,
        testing_months=testing_months,
        validation_months=validation_months,
        model_validity_months=model_validity_months,
        keep_start_fixed=keep_start_fixed,
        files_to_read=files_to_read,
        exclude_gfc=exclude_gfc,
        rolling_estimation=rolling_estimation,
    )


# %% Initialize model parameters
def init_parameters(model_type):
    search_alg_linear = "random_search"
    search_alg_trees = "random_search"
    search_alg_nets = "random_search"
    exp_tol = 0.05 / 100  # 0.05 %
    # exp_tol = 0.0

    if model_type == "ElasticNet":
        epochs = 64
        num_samples = 256
        config = {
            "hyper_params": {
                "epochs": epochs,
                "use_batches": True,
                "out_of_core": True,
                "num_samples": num_samples,
                "batch_size": tune.choice([2**12, 2**14, 2**16]),
                "search_alg": search_alg_linear,
            },
            "model_params": {
                "loss": "squared_loss",
                "penalty": "elasticnet",
                "alpha": tune.loguniform(1e-6, 1e-2),
                "l1_ratio": tune.uniform(0, 1),
                "eta0": tune.choice([1e-3, 1e-2, 1e-1]),
                "learning_rate": "invscaling",
                "verbose": 0,
                "tol": 0.0,
            },
            "optim_params": {},
            "sched_params": {},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": True,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "Lasso":
        epochs = 64
        num_samples = 256
        config = {
            "hyper_params": {
                "epochs": epochs,
                "use_batches": True,
                "out_of_core": True,
                "num_samples": num_samples,
                "batch_size": tune.choice([2**12, 2**14, 2**16]),
                "search_alg": search_alg_linear,
            },
            "model_params": {
                "loss": "squared_loss",
                "penalty": "l1",
                "alpha": tune.loguniform(1e-6, 1e-2),
                "eta0": tune.choice([1e-3, 1e-2, 1e-1]),
                "learning_rate": "invscaling",
                "verbose": 0,
                "tol": 0.0,
            },
            "optim_params": {},
            "sched_params": {},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": True,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "Ridge":
        epochs = 64
        num_samples = 256
        config = {
            "hyper_params": {
                "epochs": epochs,
                "use_batches": True,
                "out_of_core": True,
                "num_samples": num_samples,
                "batch_size": tune.choice([2**12, 2**14, 2**16]),
                "search_alg": search_alg_linear,
            },
            "model_params": {
                "loss": "squared_loss",
                "penalty": "l2",
                "alpha": tune.loguniform(1e-6, 1e-2),
                "eta0": tune.choice([1e-3, 1e-2, 1e-1]),
                "learning_rate": "invscaling",
                "verbose": 0,
                "tol": 0.0,
            },
            "optim_params": {},
            "sched_params": {},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": True,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "PLS":
        epochs = 0
        num_samples = 1
        config = {
            "hyper_params": {
                "epochs": epochs,
                "use_batches": False,
                "out_of_core": False,
                "num_samples": num_samples,
                "search_alg": "random_search",
            },
            "model_params": {"n_components": tune.grid_search([1, 2, 3, 4, 5, 6])},
            "optim_params": {},
            "sched_params": {},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": False,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "PCR":
        epochs = 0
        num_samples = 1
        config = {
            "hyper_params": {
                "epochs": epochs,
                "use_batches": False,
                "out_of_core": False,
                "num_samples": num_samples,
                "search_alg": "random_search",
            },
            "model_params": {"n_components": tune.grid_search([1, 2, 3, 4, 5, 6])},
            "optim_params": {},
            "sched_params": {},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": False,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "FFNN":
        epochs = 64
        num_samples = 256
        config = {
            "hyper_params": {
                "epochs": epochs,
                "use_batches": True,
                "out_of_core": True,
                "num_samples": num_samples,
                "n_ensemble": 8,
                "batch_size": tune.choice([2**12, 2**14, 2**16]),
                "search_alg": search_alg_nets,
            },
            "model_params": {
                "l1": tune.choice([32, 64, 128]),
                "num_hidden_layers": tune.choice([1, 2, 3, 4, 5]),
                "p": tune.uniform(0, 0.5),
            },
            "optim_params": {
                "lr": tune.choice([1e-3, 1e-2, 1e-1]),
                # "momentum": 0.9,
                "weight_decay": tune.uniform(0, 1e-1),
                # "nesterov": True,
                "amsgrad": True,
            },
            "sched_params": {"sched": "flat"},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": True,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "FFNN_test":
        epochs = 10
        num_samples = 10
        config = {
            "hyper_params": {
                "epochs": epochs,
                "use_batches": True,
                "out_of_core": True,
                "num_samples": num_samples,
                "n_ensemble": 8,
                "batch_size": 2**10,
                "search_alg": search_alg_nets,
            },
            "model_params": {
                "l1": 256,
                "num_hidden_layers": 3,
                "p": 0.2,
            },
            "optim_params": {
                "lr": 1e-3,
                "weight_decay": 1e-3,
                "amsgrad": True,
            },
            "sched_params": {"sched": "flat", "T_max": epochs},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": True,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "FFNN_SGDR":
        epochs = 64
        num_samples = 256
        config = {
            "hyper_params": {
                "epochs": epochs,
                "use_batches": True,
                "out_of_core": True,
                "num_samples": num_samples,
                "n_ensemble": 8,
                "batch_size": tune.choice([2**12, 2**14, 2**16]),
                "search_alg": search_alg_nets,
            },
            "model_params": {
                "l1": tune.choice([32, 64, 128]),
                "num_hidden_layers": tune.choice([1, 2, 3, 4, 5]),
                "p": tune.uniform(0, 0.5),
            },
            "optim_params": {
                "lr": tune.choice([1e-2, 1e-1, 1]),
                # "momentum": 0.9,
                "weight_decay": tune.uniform(0, 1e-1),
                # "nesterov": True,
                "amsgrad": True,
            },
            "sched_params": {"sched": "SGDR", "T_0": 8, "T_mult": 2},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": True,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "FFNN_SWA":
        epochs = 64
        num_samples = 256
        swa_start_pct = 0.75
        config = {
            "hyper_params": {
                "epochs": epochs,
                "swa_epochs": int(epochs * (1 - swa_start_pct)),
                "use_batches": True,
                "out_of_core": True,
                "num_samples": num_samples,
                "n_ensemble": 8,
                "batch_size": tune.choice([2**6, 2**10, 2**14]),
                "swa_start": int(epochs * swa_start_pct),
                "search_alg": search_alg_nets,
            },
            "model_params": {
                "l1": tune.choice([32, 64, 128]),
                "num_hidden_layers": tune.choice([1, 2, 3, 4, 5]),
                "p": tune.uniform(0, 0.5),
            },
            "optim_params": {
                "lr": tune.choice([1e-2, 1e-1, 1]),
                # "momentum": 0.9,
                "weight_decay": tune.uniform(0, 1e-1),
                # "nesterov": True,
                "amsgrad": True,
            },
            "sched_params": {"sched": "SGDR", "T_0": 8, "T_mult": 2},
            "swa_params": {
                "swa_lr": tune.uniform(1e-2, 1e-1),
                "anneal_epochs": 5,
                "anneal_strategy": "cos",
            },
            "tune_params": {
                "trial_stopper": False,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "LightGBR":
        epochs = 1024
        num_samples = 256
        config = {
            "hyper_params": {
                "epochs": 0,
                "use_batches": False,
                "out_of_core": False,
                "num_samples": num_samples,
                "search_alg": search_alg_trees,
            },
            "model_params": {
                "boosting_type": "gbdt",
                "objective": "regression",
                "learning_rate": tune.choice([0.01, 0.1, 1]),
                "metric": "l2",
                "num_iterations": epochs,
                "max_depth": tune.randint(2, 10),
                "num_leaves": tune.randint(2, 512),
                "min_gain_to_split": 0,
                "min_data_in_leaf": tune.choice([2**6, 2**8, 2**10, 2**12, 2**14]),
                "feature_fraction": tune.uniform(0.25, 1),
                "bagging_fraction": tune.uniform(0.25, 1),
                "bagging_freq": tune.choice([1, 10, 50]),
                "lambda_l1": tune.uniform(0, 1e-1),
                "lambda_l2": tune.uniform(0, 1e-1),
                "force_col_wise": True,  # faster to pre-select this.
            },
            "optim_params": {
                "verbose_eval": 32,
                "early_stopping_rounds": 32,
            },
            "sched_params": {},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": False,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,  # NOTE: checked that this is ok to do.
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "LightDart":
        epochs = 1024
        num_samples = 512
        config = {
            "hyper_params": {
                "epochs": 0,
                "use_batches": False,
                "out_of_core": False,
                "num_samples": num_samples,
                "search_alg": search_alg_trees,
            },
            "model_params": {
                "boosting_type": "dart",
                "objective": "regression",
                "learning_rate": tune.choice([0.01, 0.1, 1]),
                "metric": "l2",
                "num_iterations": epochs,
                "max_depth": tune.randint(2, 10),
                "num_leaves": tune.randint(2, 512),
                "min_gain_to_split": 0,
                "min_data_in_leaf": tune.choice([2**6, 2**8, 2**10, 2**12, 2**14]),
                "feature_fraction": tune.uniform(0.25, 1),
                "bagging_fraction": tune.uniform(0.25, 1),
                "bagging_freq": tune.choice([1, 10, 50]),
                "lambda_l1": tune.uniform(0, 1e-1),
                "lambda_l2": tune.uniform(0, 1e-1),
                "force_col_wise": True,  # faster to pre-select this.
                "drop_rate": tune.choice([0.05, 0.1, 0.15]),
                "skip_drop": tune.choice([0.25, 0.5]),
            },
            "optim_params": {
                "verbose_eval": 32,
                "early_stopping_rounds": 32,
            },
            "sched_params": {},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": False,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    elif model_type == "LightRF":
        epochs = 1024
        num_samples = 256
        config = {
            "hyper_params": {
                "epochs": 0,
                "use_batches": False,
                "out_of_core": False,
                "num_samples": num_samples,
                "search_alg": search_alg_trees,
            },
            "model_params": {
                "boosting_type": "rf",
                "objective": "regression",
                "learning_rate": tune.choice([0.01, 0.1, 1]),
                "metric": "l2",
                "num_iterations": epochs,
                "max_depth": tune.randint(2, 10),
                "num_leaves": tune.randint(2, 512),
                "min_gain_to_split": 0,
                "min_data_in_leaf": tune.choice([2**6, 2**8, 2**10, 2**12, 2**14]),
                "feature_fraction": tune.uniform(0.25, 1),
                "bagging_fraction": tune.uniform(0.25, 1),
                "bagging_freq": tune.choice([1, 10, 50]),
                "lambda_l1": tune.uniform(0, 1e-1),
                "lambda_l2": tune.uniform(0, 1e-1),
                "force_col_wise": True,  # faster to pre-select this.
            },
            "optim_params": {
                "verbose_eval": 32,
                "early_stopping_rounds": 32,
            },
            "sched_params": {},
            "swa_params": {},
            "tune_params": {
                "trial_stopper": False,
                "trial_num_results": 8,
                "trial_grace_period": 0,
                "trial_tol": 0.0,
                "experiment_stopper": False,
                "exp_top_models": 10,
                "exp_num_results": num_samples // 4,
                "exp_grace_period": num_samples // 4,
                "exp_tol": exp_tol,
            },
        }

    else:
        raise ValueError("Wrong model type selected. Selection: %s" % model_type)

    return config


# %%
def init_models(model_type, params: dict = {}):
    if model_type == "ElasticNet":
        model = SGDRegressor(
            **params,
        )

    elif model_type == "Lasso":
        model = SGDRegressor(
            **params,
        )

    elif model_type == "Ridge":
        model = SGDRegressor(
            **params,
        )

    elif model_type == "PLS":
        model = PLSRegression(**params)

    elif model_type == "PCR":
        model = Pipeline([("pca", PCA(**params)), ("regressor", LinearRegression())])

    elif model_type in ["FFNN", "FFNN_test", "FFNN_SWA", "FFNN_SGDR"]:
        model = FFNN(**params)

    elif model_type in ["LightGBR", "LightDart", "LightRF"]:
        model = lgbm.LGBMRegressor(**params)

    else:
        raise ValueError("Wrong model type selected.")

    return model


# %%
