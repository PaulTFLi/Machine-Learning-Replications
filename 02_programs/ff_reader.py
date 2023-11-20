# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
"""
Reads in data from Kenneth French's website.

I wrote this simple script, because the pandas_datareader
package fails for Operating Profitability datasets.
Apparently the structure has been changed over time.

To get a list of available datasets, call "get_datasets"
which calls the same function from the pandas_datareader
package.
"""

# %%
import pandas as pd
import numpy as np
from pandas_datareader.famafrench import get_available_datasets


# %%
def read_dataset(dataset_name, verbose=False):
    """Reads Fama/French datasets."""

    # get the number of rows to skip.
    read = False
    skipped_rows = 0
    while read is False:
        try:
            data = pd.read_csv(
                ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" + dataset_name + "_CSV.zip"),
                skiprows=skipped_rows,
                index_col=0,
                nrows=1000,
            )
            read = True
        except (pd.errors.ParserError, IndexError, UnicodeDecodeError) as e:
            if verbose:
                print(e, ", skipping an additional row.")
            skipped_rows += 1

    # get row of first "real" date:
    if (data.index.dtype) == "object":
        dates = data.index.str.extract(r"^(\d{8}|\d{6})", expand=False)
        ii_first_date = np.where(~dates.isnull())[0][0]
        if ii_first_date > 0:
            ii_first_date += 1
    else:
        ii_first_date = 0

    # read in data:
    data = pd.read_csv(
        ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" + dataset_name + "_CSV.zip"),
        skiprows=ii_first_date + skipped_rows,
        index_col=0,
    )

    # skip annual data at the bottom:
    annual_data_cutoff = np.where(data.isnull().all(axis=1))[0]
    if len(annual_data_cutoff > 0):
        data = data.iloc[: annual_data_cutoff[0]]

    # check type of data: monthly vs. daily:
    if "daily" in dataset_name:
        data.index = pd.to_datetime(data.index, format="%Y%m%d")
    else:
        data.index = pd.to_datetime(data.index, format="%Y%m").to_period("M")

    # data is given in percentage points:
    data = data.astype("f8")
    data /= 100

    return data


# %%
def get_datasets():
    return get_available_datasets()


# %%
