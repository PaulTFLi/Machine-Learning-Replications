# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

import pandas as pd
import numpy as np

# %% Institutional ownership
insti = pd.read_parquet("../../03_data/sec13f.parquet")
insti.rdate = pd.to_datetime(insti.rdate, format="%Y%m%d")
insti.fdate = pd.to_datetime(insti.fdate, format="%Y%m%d")
insti["yq"] = insti.rdate.dt.to_period("Q")

# duplicates = insti.groupby(["rdate", "cusip", "cik"]).apply(lambda x: x.shape[0])

crsp = pd.read_parquet(
    "../../03_data/crsp_prices_daily.pq",
    columns=["permno", "date", "close", "shrout", "ncusip", "cusip"],
)
crsp = crsp.sort_values("date")
crsp["yq"] = crsp.date.dt.to_period("Q")
crsp = crsp.groupby(["permno", "yq"]).last()
crsp.reset_index(inplace=True)

cusip_match = pd.merge(insti, crsp, left_on=["yq", "cusip"], right_on=["yq", "cusip"], how="left")
cusip_matched = cusip_match.loc[~np.isnan(cusip_match.close), :]
cusip_matched["cusip_crsp"] = cusip_matched.cusip
cusip_nonmatched = cusip_match.loc[np.isnan(cusip_match.close), :]

cusip_nonmatched = cusip_nonmatched[insti.columns.values]
ncusip_match = pd.merge(
    cusip_nonmatched, crsp, how="left", left_on=["cusip", "yq"], right_on=["ncusip", "yq"]
)
ncusip_matched = ncusip_match.loc[~np.isnan(ncusip_match.close), :]
ncusip_nonmatched = ncusip_match.loc[np.isnan(ncusip_match.close), :]

ncusip_matched.rename({"cusip_x": "cusip", "cusip_y": "cusip_crsp"}, inplace=True, axis=1)

insti = pd.concat([cusip_matched, ncusip_matched])

del ncusip_nonmatched, ncusip_matched, ncusip_match, cusip_nonmatched, cusip_matched, cusip_match

insti.shrout *= 1000
insti["share"] = insti.shares / insti.shrout

# erase all obs with single-firm share greater than 50%
insti = insti.loc[insti.share < 0.5, :]

insti_share = insti.groupby(["permno", "rdate"]).agg({"share": np.nansum})
insti_share.reset_index(inplace=True)
insti_share_last = insti_share.loc[insti_share.rdate == insti_share.rdate.max(), :].copy()
insti_share_last.share = np.nan
insti_share_last.rdate = insti_share_last.rdate + pd.DateOffset(months=3)
insti_share = pd.concat([insti_share, insti_share_last])

insti_share["date"] = insti_share.rdate + pd.offsets.MonthBegin(1)
insti_share.sort_values("date", inplace=True)
insti_share.set_index("date", inplace=True)
insti_share.drop(columns="rdate", inplace=True)
insti_share = insti_share.groupby("permno").resample("D").ffill()
insti_share.drop(columns="permno", inplace=True)
insti_share.reset_index(inplace=True)
insti_share.permno = insti_share.permno.astype(int)
insti_share.dropna(inplace=True)

insti_share.to_parquet("../../03_data/institutional_holdings_13f.pq")


# %%
