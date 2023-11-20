# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
from typing import Tuple, Union
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import re
import glob
import os
from random import shuffle
import lightgbm as lgbm
from cross_validation_time_series import scoring_table_sample_split
from init_model_parameters import init_sample
import gc


# %%
"""
NOTE: currently only works for option sample
"""


def create_lightgbm_dataset(
    fileLoc: str,
    concatLoc: str,
    target_column: str,
    feature_groups: str,
    rolling_estimation: bool = False,
    exclude_gfc: bool = False,
):

    os.makedirs(concatLoc, exist_ok=True)

    sample_setup = init_sample(
        fileLoc,
        target_column,
        feature_groups=feature_groups,
        rolling_estimation=rolling_estimation,
        exclude_gfc=exclude_gfc,
    )
    date_column = sample_setup["date_column"]
    other_columns_to_delete = sample_setup["other_columns_to_delete"]
    initial_training_months = sample_setup["initial_training_months"]
    keep_start_fixed = sample_setup["keep_start_fixed"]
    validation_months = sample_setup["validation_months"]
    testing_months = sample_setup["testing_months"]
    model_validity_months = sample_setup["model_validity_months"]

    files_to_read = [
        os.path.abspath(f) for f in glob.glob(fileLoc + "/characteristics_*.pq") if re.match(r".*([1-2][0-9]{3})", f)
    ]

    data = pd.concat([pd.read_parquet(f, columns=["date_col"]) for f in files_to_read])
    _, results_table = scoring_table_sample_split(
        data,
        date_column,
        initial_training_months,
        validation_months,
        testing_months,
        model_validity_months,
        keep_start_fixed,
    )
    print(results_table)

    print("Reading file bounds.", flush=True)
    dates_in_files = {}
    for f in files_to_read:
        data = pd.read_parquet(f, columns=[date_column]).squeeze()
        dates_in_files[f] = [data.min(), data.max()]
    del data
    dates_in_files = pd.DataFrame(dates_in_files, index=["start", "end"]).T

    for ith_step, (s_train, e_train, s_val, e_val) in enumerate(
        results_table[["train_start", "train_end", "validate_start", "validate_end"]].to_numpy()
    ):
        print("Working on step i=%d" % ith_step)
        print("Reading raw data.")
        files = dates_in_files.index[(dates_in_files["start"] >= s_train) & (dates_in_files["end"] <= e_val)].tolist()
        files.sort()
        X, y = read_data_parallel(
            files_to_read=files,
            date_column=date_column,
            target_column=target_column,
            other_columns_to_delete=other_columns_to_delete,
            start=s_train,
            end=e_val,
            n_jobs=12,
        )

        print("Splitting")
        if exclude_gfc:
            X_val = X[s_val:e_val]
            X_val = X_val[~((X_val.index >= "2007-01-01") & (X_val.index < "2010-01-01"))]
            X_val = X_val.to_numpy()
            y_val = y[s_val:e_val]
            y_val = y_val[~((y_val.index >= "2007-01-01") & (y_val.index < "2010-01-01"))]
            y_val = y_val.to_numpy()

            X = X[s_train:e_train]
            X = X[~((X.index >= "2007-01-01") & (X.index < "2010-01-01"))]
            print(X.index.unique())
            X = X.to_numpy()
            y = y[s_train:e_train]
            y = y[~((y.index >= "2007-01-01") & (y.index < "2010-01-01"))]
            y = y.to_numpy()
        else:
            X_val = X[s_val:e_val].to_numpy()
            y_val = y[s_val:e_val].to_numpy()
            X = X[s_train:e_train].to_numpy()
            y = y[s_train:e_train].to_numpy()

        print("Creating dataset")
        train_data = lgbm.Dataset(X, label=y)
        val_data = train_data.create_valid(X_val, label=y_val)
        train_path = os.path.join(concatLoc, "%d_train_%s.bin" % (ith_step, feature_groups))
        if os.path.isfile(train_path):
            os.remove(train_path)
        train_data.save_binary(train_path)
        val_path = os.path.join(concatLoc, "%d_val_%s.bin" % (ith_step, feature_groups))
        if os.path.isfile(val_path):
            os.remove(val_path)
        val_data.save_binary(val_path)

        del X_val, y_val, val_data, X, y, train_data
        gc.collect()


# %% File reading capabilities.
def read_data_single_file(
    file,
    date_column: str,
    target_column: str,
    other_columns_to_delete: list,
    start=None,
    end=None,
):
    # ----- read in testing data -----
    if isinstance(file, list):
        file = file[0]
    X = pd.read_parquet(file)
    col_to_delete = [target_column, date_column] + other_columns_to_delete
    X = X.dropna(subset=[target_column])
    y = X[target_column]
    y = y.astype("f4")  # single precision.
    if start and end:
        X = X.loc[start:end]
        y = y.loc[start:end]
    X = X.drop(columns=col_to_delete)
    return X, y


def read_data_parallel(
    files_to_read: str,
    date_column: str,
    target_column: str,
    other_columns_to_delete: list,
    start: str = None,
    end: str = None,
    n_jobs: int = 1,
):
    X = Parallel(n_jobs=min(len(files_to_read), n_jobs), verbose=True, backend="loky")(
        delayed(pd.read_parquet)(fn) for fn in files_to_read
    )
    X = pd.concat(X)
    X = X.dropna(subset=[target_column])
    X = X.sort_index()
    y = X[target_column]
    y = y.astype("f4")  # single precision.
    col_to_delete = [target_column, "permno", date_column] + other_columns_to_delete
    if start and end:
        X = X.loc[start:end]
        y = y.loc[start:end]
    X = X.drop(columns=col_to_delete)
    return X, y


# %% Dataloader capability.
def DataLoader(
    files_to_read: Union[list, str],
    out_of_core: bool,
    date_column: str,
    target_column: str,
    other_columns_to_delete: list,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    batch_size: int = 0,
    file_batch_size: int = 1,
    shuffle_data: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dataloader returning pd.DataFrame/pd.Series tuples or np.ndarray tuples.

    NOTE: this is designed to read the data from disk **every epoch**!
        To avoid this, read the entire data in once, before training,
        then iterate over batches using DataIterator below.

    Args:
        files_to_read (Union[list, str]): File paths to read in.
            Can either be a str (or list of len() = 1), in which case the combined file
            is read. This assumes that data fits in memory. This is then batched over.
        out_of_core (bool): Whether to perform out of core computation.
        date_column (str): Column with dates. Dropped from return values.
        target_column (str): Target column (i.e. returns) used for y values, dropped from X.
        other_columns_to_delete (list): Other columns to delete (i.e. permno). Dropped from
            return values.
        start (Union[str, pd.Timestamp]): Start date.
        end (stUnion[str, pd.Timestamp]): End date.
        batch_size (int): Size of each mini batch.
        file_batch_size (int, optional): How many files to read in memory at once.
            Defaults to 1.
        shuffle (bool, optional): Whether to shuffle data. Defaults to False.
            Note that this shuffles the entire dataset when it is read in memory at once
            (i.e. for one file in <files_to_read>), and shuffles the files + subdatasets
            when performing out-of-core computation.

    Yields:
        Tuple(np.ndarray, np.ndarray): X, y (input, response) pairs per batch.
    """
    batch_size = int(batch_size)

    # if only one file is given in a list, extract the file path:
    if out_of_core:
        """reads files iteratively."""
        if file_batch_size > 1:
            if shuffle_data:  # shuffle files for file batches
                shuffle(files_to_read)
            for file_batch in np.arange(0, len(files_to_read), file_batch_size, dtype="i4"):
                X, y = read_data_parallel(
                    files_to_read[file_batch : file_batch + file_batch_size],
                    date_column,
                    target_column,
                    other_columns_to_delete,
                    start,
                    end,
                    file_batch_size,
                )
                X = X.to_numpy()
                y = y.to_numpy()
                if shuffle_data:  # shuffle values in each file
                    p = np.random.permutation(len(y))
                    y = y[p]
                    X = X[p]
                    del p
                if batch_size < 1:
                    batch_size = len(X)
                for batch in np.arange(0, len(X), batch_size, dtype="i4"):
                    if ((len(X) - batch) < batch_size) & (file_batch < (len(files_to_read) - file_batch_size)):
                        X_tmp = X[batch : batch + batch_size]
                        y_tmp = y[batch : batch + batch_size]
                    else:
                        if len(X) - batch != 1:
                            yield X[batch : batch + batch_size], y[batch : batch + batch_size]
        else:
            if shuffle_data:  # shuffle files
                shuffle(files_to_read)

            X_tmp = []
            y_tmp = []
            for i, file in enumerate(files_to_read):
                X, y = read_data_single_file(
                    file,
                    date_column,
                    target_column,
                    other_columns_to_delete,
                    start,
                    end,
                )
                X = X.to_numpy()
                y = y.to_numpy()
                if shuffle_data:  # shuffle values in each file
                    p = np.random.permutation(len(y))
                    y = y[p]
                    X = X[p]
                    del p
                if len(X_tmp) > 0:
                    X = np.concatenate((X_tmp, X))
                    y = np.concatenate((y_tmp, y))
                if batch_size < 1:
                    batch_size = len(X)
                for batch in np.arange(0, len(X), batch_size, dtype="i4"):
                    if ((len(X) - batch) < batch_size) & (i < (len(files_to_read) - 1)):
                        X_tmp = X[batch : batch + batch_size]
                        y_tmp = y[batch : batch + batch_size]
                    else:
                        if len(X) - batch != 1:
                            yield X[batch : batch + batch_size], y[batch : batch + batch_size]

    else:
        """
        read all files and potentially batch over dataset.
        This most closely resembles what pytorch.DataLoader would
        do.
        """
        if batch_size < 1:
            X, y = read_data_parallel(
                files_to_read=files_to_read,
                date_column=date_column,
                target_column=target_column,
                other_columns_to_delete=other_columns_to_delete,
                start=start,
                end=end,
            )
            X = X.to_numpy()
            y = y.to_numpy()
            if shuffle_data:
                p = np.random.permutation(len(y))
                y = y[p]
                X = X[p]
            yield X, y

        else:
            X, y = read_data_parallel(
                files_to_read=files_to_read,
                date_column=date_column,
                target_column=target_column,
                other_columns_to_delete=other_columns_to_delete,
                start=start,
                end=end,
            )
            X = X.to_numpy()
            y = y.to_numpy()
            if shuffle_data:
                p = np.random.permutation(len(y))
                y = y[p]
                X = X[p]
            for batch in np.arange(0, len(X), batch_size, dtype="i4"):
                if len(X) - batch != 1:
                    yield X[batch : batch + batch_size], y[batch : batch + batch_size]


def DataIterator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 0,
    shuffle_data: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterate over X, y pairs already in memory.
    Returns the selected batch size and potentially shuffles
    the data.
    """

    batch_size = int(batch_size)
    if shuffle_data:
        p = np.random.permutation(len(y))
        y = y[p]
        X = X[p]
    if batch_size < 1:
        yield X, y
    else:
        for batch in np.arange(0, len(X), batch_size, dtype="i4"):
            if len(X) - batch != 1:
                yield X[batch : batch + batch_size], y[batch : batch + batch_size]


# %%
