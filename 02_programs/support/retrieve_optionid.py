# %%

from joblib.parallel import cpu_count
import pandas as pd
import glob
import os
from joblib import Parallel, delayed


# %%
# folder = "../../03_data/"
folder = "/cloud/wwu1/d_taqopt/taq/_projects/h_beck18/options_ml/03_data/"
files = glob.glob(os.path.join(folder, "prices", "*.pq"))

optionid_links = []


def reader(file):
    data = pd.read_parquet(
        file, columns=["date", "exdate", "cp_flag", "strike_price", "optionid", "secid"]
    )
    data["date"] = pd.to_datetime(data.date, format="%Y-%m-%d")
    data["exdate"] = pd.to_datetime(data.exdate, format="%Y-%m-%d")
    data["cp_flag"] = data["cp_flag"].str.lower()
    data["cp_flag"] = data["cp_flag"].replace("c", 1).replace("p", -1).astype("i4")
    data["strike_price"] /= 1000
    data = data.drop_duplicates(subset=["optionid", "exdate", "cp_flag", "strike_price"])
    data["optionid"] = data["optionid"].astype("i4")
    return data


optionid_links = Parallel(n_jobs=cpu_count() // 2, verbose=True)(
    delayed(reader)(file) for file in files
)
optionid_links = pd.concat(optionid_links)

# map permno and ticker to secid:
crsp_link = pd.read_parquet(
    os.path.join(folder, "mathis_linking_suite_crsp_optionmetrics.pq"),
    columns=["date", "permno", "secid", "shrout", "close"],
)
crsp_link = crsp_link.dropna()
crsp_link = crsp_link[["date", "permno", "secid"]]

optionid_links = optionid_links.merge(crsp_link, on=["date", "secid"])

# map tickers:
crsp_permno_ticker_map = pd.read_parquet(
    os.path.join(folder, "crsp_prices_daily.pq"), columns=["permno", "date", "ticker"]
)
optionid_links = optionid_links.merge(crsp_permno_ticker_map, on=["date", "permno"])

# saving:
optionid_links = optionid_links.drop(columns=["date"])
optionid_links.to_parquet("../../03_data/optionid_links.pq")


# %%
