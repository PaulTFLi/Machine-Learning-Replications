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
from pandas.tseries.offsets import MonthEnd


# %%
def scoring_table_sample_split(
    data,
    date_column,
    initial_training_months,
    validation_months,
    testing_months,
    model_validity_months,
    keep_start_fixed: bool = True,
):
    months_in_sample = data.groupby(date_column).size()
    # months_in_sample = months_in_sample.append(pd.Series({months_in_sample.index.min() - pd.DateOffset(months=1): 0}))
    months_in_sample = months_in_sample.append(pd.Series({pd.Timestamp(1996, 1, 1): 0}))
    months_in_sample = months_in_sample.sort_index()

    months = months_in_sample.index.to_period("M").unique()  # .astype("str")
    # num_steps = (len(months) - initial_training_months - validation_months - testing_months + 1) / model_validity_months
    num_steps = (len(months) - initial_training_months - validation_months - testing_months) / model_validity_months
    scoring_table = pd.DataFrame(
        columns=[
            "train_start",
            "train_end",
            "validate_start",
            "validate_end",
            "test_start",
            "test_end",
            "train_val_size",
            "largest_train_size",
        ],
    )
    scoring_table.index.name = "ith_step"
    for i in np.arange(num_steps + 1, dtype="int"):
        # ----- split data in training/validation/testing:
        if keep_start_fixed:
            init = 0
        else:
            init = i
        train_start = months[init * model_validity_months].to_timestamp().strftime("%Y-%m-%d")
        train_end = (
            months[i * model_validity_months + initial_training_months - 1].to_timestamp() + MonthEnd()
        ).strftime("%Y-%m-%d")
        validate_start = months[i * model_validity_months + initial_training_months].to_timestamp().strftime("%Y-%m-%d")
        validate_end = (
            months[i * model_validity_months + initial_training_months + validation_months - 1].to_timestamp()
            + MonthEnd()
        ).strftime("%Y-%m-%d")
        test_start = (
            months[i * model_validity_months + initial_training_months + validation_months]
            .to_timestamp()
            .strftime("%Y-%m-%d")
        )
        test_end = (
            months[
                np.minimum(
                    i * model_validity_months + initial_training_months + validation_months + testing_months - 1,
                    len(months) - 1,
                )
            ].to_timestamp()
            + MonthEnd()
        ).strftime("%Y-%m-%d")

        scoring_table = scoring_table.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "validate_start": validate_start,
                "validate_end": validate_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_val_size": months_in_sample.loc[train_start:train_end].sum(),
                "largest_train_size": months_in_sample.loc[train_start:train_end].max(),
            },
            ignore_index=True,
        )

    scoring_table["done"] = 0
    scoring_table["trials_exist"] = 0

    scoring_table = scoring_table.iloc[::-1]
    scoring_table = scoring_table.reset_index(drop=True)

    return months_in_sample, scoring_table


# %%
