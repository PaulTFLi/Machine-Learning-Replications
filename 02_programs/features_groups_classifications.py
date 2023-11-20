# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
""" Features, feature groups and classification variables.

Returns a Dataframe of possible features and their resp. feature groups
as well as a dictionary of possible classification splits.

Also reports the features as a table to .tex.

Includes the following analyses:
    - Sum2: characteristics table.
"""

# %%
import pandas as pd
import os
import numpy as np
from numba import njit
import glob
from init_model_parameters import possible_samples


# %%
# create volume dummies on a permno basis:
def quantile_cuts(data, col, n_port, port_names: list):
    qs = data.groupby("date")[col].apply(lambda x: pd.qcut(x, n_port, labels=False, duplicates="drop"))
    names = {}
    for i, name in enumerate(port_names):
        names[i] = name
    qs = qs.fillna("no_data")
    qs = qs.replace(names)
    return qs


@njit
def vw_mean(target, weight):
    if weight.sum() == 0:
        return np.nan
    else:
        return (target * weight).sum() / weight.sum()


def return_option_flows(data_for_flows: pd.DataFrame, volume_type: str, investor_groups: list):
    flows = data_for_flows[
        [
            c
            for c in data_for_flows.columns
            if c.endswith("_" + volume_type) and any([c.startswith(group) for group in investor_groups])
        ]
    ].sum(axis=1)
    return flows


# %% Hyperparameter
print("Possible samples:")
# for i, elem in enumerate(possible_samples):
#     print("%d: %s" % (i, elem))
# fileLoc = input("Which file location?   ")
fileLoc = 1
fileLoc = possible_samples[int(fileLoc)]

os.makedirs(os.path.join(fileLoc, "analysis"), exist_ok=True)

# %% Classification groups:
classifier_files = glob.glob(os.path.join(fileLoc, "classifier*"))
class_groups = []
for file in classifier_files:
    print("Working on file {}.".format(file))
    class_groups.append(pd.read_parquet(file))
class_groups = pd.concat(class_groups)
class_groups = class_groups.reset_index(drop=True)
class_groups["optionid"] = class_groups["optionid"].astype("i8")
class_groups["permno"] = class_groups["permno"].astype("i4")

print(class_groups.dtypes)
print(class_groups["date"])

try:
    class_groups["month"] = class_groups["date"].dt.to_period("M")
except AttributeError:
    class_groups["month"] = class_groups["date"]


# ---- option-based stuff
class_groups["type"] = "call"
class_groups.loc[class_groups["P"] == 1, "type"] = "put"

# ---- stock based stuff
group_cols = ["baspread", "beta", "bm", "idiovol", "lev", "ivrv", "mom12m", "mve", "ill", "iv"]

# ---- stock illiquidity:
per_permno_ill = class_groups.groupby(["date", "permno"]).baspread.first()
per_permno_ill = per_permno_ill.groupby("date").apply(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
per_permno_ill.name = "split_stock_liquidity"
class_groups = class_groups.merge(per_permno_ill, on=["date", "permno"], how="left")
class_groups["split_stock_liquidity"] = class_groups["split_stock_liquidity"].astype("str")
class_groups["split_stock_liquidity"] = class_groups["split_stock_liquidity"].fillna("no_data")


# ---- option illiquidty:
class_groups["split_option_liquidity"] = class_groups.groupby("date")["optspread"].apply(
    lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
)
class_groups["split_option_liquidity"] = class_groups["split_option_liquidity"].astype("str")
class_groups["split_option_liquidity"] = class_groups["split_option_liquidity"].fillna("no_data")


per_permno_ill = (
    class_groups[class_groups.bucket == "short_term_atm"]
    .groupby(["date", "permno"])
    .apply(lambda x: vw_mean(x["optspread"].to_numpy(), x["oi"].to_numpy()))
)
per_permno_ill = per_permno_ill.groupby("date").apply(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
per_permno_ill.name = "split_option_liquidity_permno"
class_groups = class_groups.merge(per_permno_ill, on=["date", "permno"], how="left")
class_groups["split_option_liquidity_permno"] = class_groups["split_option_liquidity_permno"].astype("str")
class_groups["split_option_liquidity_permno"] = class_groups["split_option_liquidity_permno"].fillna("no_data")


# ---- institutional holdings (13-F):
inst_hld = pd.read_parquet("../03_data/institutional_holdings_13f.pq")
inst_hld.columns = ["permno", "date", "inst_share"]
inst_hld = inst_hld.drop_duplicates()
inst_hld.date = inst_hld.date.dt.to_period("M")
inst_hld = inst_hld.groupby(["date", "permno"]).last().reset_index()
inst_hld.to_parquet(os.path.join(fileLoc, "analysis/institutional_holdings.pq"))
inst_hld = inst_hld.rename(columns={"date": "month"})

class_groups = class_groups.merge(inst_hld, on=["month", "permno"], how="left")

col = "inst_share"
col_median = class_groups.groupby("date")[col].transform("median")
class_groups["split_" + col] = "high_" + col
class_groups.loc[class_groups[col] <= col_median, "split_" + col] = "low_" + col
class_groups.loc[class_groups[col].isnull(), "split_" + col] = "no_data"

class_groups["split_inst_share_quintile"] = class_groups.groupby("date")["inst_share"].apply(
    lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
)
class_groups["split_inst_share_quintile"] = class_groups["split_inst_share_quintile"].astype("str")
class_groups["split_inst_share_quintile"] = class_groups["split_inst_share_quintile"].fillna("no_data")

# ---- analyst coverage:
analysts = pd.read_csv("../03_data/analysts.csv")
analysts = analysts.dropna(subset=["permno"])
analysts["datadate"] = pd.to_datetime(analysts["datadate"], format="%Y%m%d")
analysts = analysts[["permno", "fyear", "num_analysts"]]
analysts.to_parquet(os.path.join(fileLoc, "analysis/analyst_coverage.pq"))

class_groups["year"] = class_groups["date"].dt.year
class_groups = class_groups.merge(analysts, left_on=["permno", "year"], right_on=["permno", "fyear"], how="left")

class_groups["greater_median"] = class_groups.groupby("fyear").num_analysts.transform("median")
class_groups["greater_median"] = (class_groups["num_analysts"] > class_groups["greater_median"]).astype("i4")

class_groups["split_analyst_coverage"] = "no_data"
class_groups.loc[class_groups["greater_median"] == 1, "split_analyst_coverage"] = "analyst_coverage"
class_groups.loc[class_groups["greater_median"] == 0, "split_analyst_coverage"] = "no_analyst_coverage"
class_groups.loc[class_groups["num_analysts"].isnull(), "split_analyst_coverage"] = "no_data"
del analysts

# quintiles:
class_groups["split_analyst_quintile"] = class_groups.groupby("date")["num_analysts"].apply(
    lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
)
class_groups["split_analyst_quintile"] = class_groups["split_analyst_quintile"].astype("str")
class_groups["split_analyst_quintile"] = class_groups["split_analyst_quintile"].fillna("no_data")


# ---- option volume retail vs. institutional:
flow_files = glob.glob("../03_data/open_close/*_merge.pq")
option_flows = [pd.read_parquet(f) for f in flow_files]
option_flows = pd.concat(option_flows)

for investor in ["firm", "bd", "prop", "cust", "pcust"]:
    cols = [c for c in option_flows.columns if c.startswith(investor + "_")]
    vol = option_flows[cols].sum(axis=1)
    vol.name = investor + "_vol"
    buys = option_flows[[c for c in cols if c.endswith("_buys")]].sum(axis=1)
    buys.name = investor + "_buys"
    sells = option_flows[[c for c in cols if c.endswith("_sells")]].sum(axis=1)
    sells.name = investor + "_sells"
    svol = buys - sells
    svol.name = investor + "_svol"
    option_flows = option_flows.drop(columns=cols)
    option_flows = pd.concat((option_flows, vol, buys, sells, svol), axis=1)

option_flows = option_flows.rename(columns={"ym": "month"})
option_flows = option_flows.drop(columns=["expiry", "strike", "type", "root", "sec_type", "strike_price", "secid"])


# ---- per contract stuff:
flows = option_flows[["month", "optionid"]]
flows["prof"] = return_option_flows(option_flows, "svol", ["prop", "firm", "pcust"]) / return_option_flows(
    option_flows, "vol", ["cust", "prop", "firm", "pcust"]
)
flows["prof"] = flows["prof"].replace([-np.inf, np.inf, np.nan], 0)

flows["cust"] = return_option_flows(option_flows, "svol", ["cust"]) / return_option_flows(
    option_flows, "vol", ["cust", "prop", "firm", "pcust"]
)
flows["cust"] = flows["cust"].replace([-np.inf, np.inf, np.nan], 0)

# merge:
class_groups = class_groups.merge(flows, on=["month", "optionid"], how="left")
class_groups["split_prof_q"] = quantile_cuts(
    class_groups, "prof", 3, [r"$V^I < 0$", r"$V^I \approx 0$", r"$V^I > 0$"]
).values
class_groups["split_cust_q"] = quantile_cuts(
    class_groups, "cust", 3, [r"$V^R < 0$", r"$V^R \approx 0$", r"$V^R > 0$"]
).values
class_groups["split_joint_q"] = class_groups["split_prof_q"] + "_" + class_groups["split_cust_q"]
class_groups["split_joint_q"] = class_groups["split_joint_q"].replace("no_data_no_data", "no_data")
# ----


# ---- per permno stuff:
option_flows_permno = option_flows.groupby(["month", "permno"]).sum()
option_flows_permno = option_flows_permno.reset_index()
option_flows_permno = option_flows_permno.drop(columns=["optionid"])
flows = option_flows_permno[["month", "permno"]]
flows["prof"] = return_option_flows(option_flows_permno, "svol", ["prop", "firm", "pcust"]) / return_option_flows(
    option_flows_permno, "vol", ["cust", "prop", "firm", "pcust"]
)
flows["prof"] = flows["prof"].replace([-np.inf, np.inf, np.nan], 0)

flows["cust"] = return_option_flows(option_flows_permno, "svol", ["cust"]) / return_option_flows(
    option_flows_permno, "vol", ["cust", "prop", "firm", "pcust"]
)
flows["cust"] = flows["cust"].replace([-np.inf, np.inf, np.nan], 0)
flows.columns = [c + "_permno" if c not in ["month", "permno"] else c for c in flows.columns]

# merge:
class_groups = class_groups.merge(flows, on=["month", "permno"], how="left")
class_groups["split_prof_q_permno"] = quantile_cuts(
    class_groups, "prof_permno", 3, [r"$V^I < 0$", r"$V^I \approx 0$", r"$V^I > 0$"]
).values
class_groups["split_cust_q_permno"] = quantile_cuts(
    class_groups, "cust_permno", 3, [r"$V^R < 0$", r"$V^R \approx 0$", r"$V^R > 0$"]
).values
class_groups["split_joint_q_permno"] = class_groups["split_prof_q_permno"] + "_" + class_groups["split_cust_q_permno"]
class_groups["split_joint_q_permno"] = class_groups["split_joint_q_permno"].replace("no_data_no_data", "no_data")
# ----

option_flows = option_flows.reset_index(drop=True)
option_flows.to_parquet(os.path.join(fileLoc, "analysis/option_flows_with_info.pq"))

# %% ---- save
class_groups.to_parquet(os.path.join(fileLoc, "analysis/class_groups_and_info.pq"))

to_keep = []
for col in class_groups.columns:
    if col.startswith("return_") | col.startswith("gain_"):
        to_keep.append(col)
    elif col.startswith("median_") | col.startswith("quantile_") | col.startswith("split_"):
        to_keep.append(col)
    elif col in ["date", "permno", "optionid", "bucket", "type"]:
        to_keep.append(col)
print(to_keep)
class_groups = class_groups[to_keep]
class_groups.to_parquet(os.path.join(fileLoc, "analysis/class_groups.pq"))


# %%
