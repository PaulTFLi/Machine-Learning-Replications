# %%
# Packages
import pandas as pd
import numpy as np


# %%
# Option characteristics
N = 500  # options
Ts = pd.date_range("1996-02-01", end="2020-12-01", freq="M")
T = len(Ts)  # months
feature_names = pd.read_excel("../04_results/option_sample/analysis/features.xlsx")["feature"]

features = pd.DataFrame(
    np.random.uniform(low=-0.5, high=0.5, size=(T * N, len(feature_names))), columns=feature_names.values
)
features["date"] = np.repeat(Ts, N).to_period("M").to_timestamp()
features["optionid"] = np.tile(["O_" + str(i) for i in np.arange(0, N)], T)
features = features.set_index(["date", "optionid"])
features["return_dh_daily_inv"] = features.abs().mean(axis=1) - 0.25

# save by month
for date in features.index.get_level_values("date").unique():
    tmp = features.loc[date]
    tmp["date"] = date
    tmp = tmp.reset_index().set_index(["date", "optionid"])
    tmp.to_parquet(
        f"../04_results/option_sample/characteristics_{str(date.month).zfill(2)}_{str(date.year).zfill(4)}.pq"
    )


# --- Option classifiers
classifier = features[
    [
        "baspread",
        "beta",
        "bm",
        "idiovol",
        "lev",
        "ivrv",
        "mom12m",
        "mve",
        "ill",
        "close",
        "ttm",
        "moneyness",
        "m_degree",
        "mid",
        "iv",
        "delta",
        "C",
        "P",
        "gamma",
        "vega",
        "theta",
        "oi",
        "optspread",
        "embedlev",
        "doi",
    ]
]

# save by month
for date in classifier.index.get_level_values("date").unique():
    tmp = classifier.loc[date]
    tmp["date"] = date
    tmp = tmp.reset_index().set_index(["date", "optionid"])
    tmp.to_parquet(
        f"../04_results/option_sample/characteristics_{str(date.month).zfill(2)}_{str(date.year).zfill(4)}.pq"
    )


# %%
# CRSP market return
Ts = pd.date_range("1996-02-01", end="2020-12-01", freq="M")
T = len(Ts)  # months
crsp = pd.DataFrame(np.random.randn(T), columns=["vwretx"], index=Ts)
crsp.index.name = "date"
crsp.to_parquet("../03_data/crsp_market_returns.csv")


# %%
# Yield curve
Ts = pd.date_range("1996-02-01", end="2020-12-01", freq="M")
T = len(Ts)  # months
yields = pd.DataFrame(np.random.randn(T), columns=["rate"], index=Ts)
yields["days"] = 10
yields.index.name = "date"
yields = yields.reset_index()
yields.to_parquet("../03_data/yield_curve.zip")


# %%
# Firm-level events
Ts = pd.date_range("1996-02-01", end="2020-12-01", freq="M")
T = len(Ts)  # months
events = pd.DataFrame(
    np.random.randint(low=0, high=2, size=(T, 6)),
    columns=["distribution", "earnings", "news", "fomc", "great_financial_crisis", "recession"],
    index=Ts,
)
events["permno"] = np.random.randint(low=10000, high=50000, size=T)
events.index.name = "date"
events = events.set_index("permno", append=True)
events.to_parquet("../03_data/firm_level_events.pq")


# %%
# Analysts
Ts = pd.date_range("1996-02-01", end="2020-12-01", freq="M")
T = len(Ts)  # months
analysts = pd.DataFrame(
    np.random.randint(low=0, high=10, size=T),
    columns=["num_analysts"],
    index=Ts,
)
analysts["permno"] = np.random.randint(low=10000, high=50000, size=T)
analysts.index.name = "fyear"
analysts.index = analysts.index.year
analysts = analysts.set_index("permno", append=True)
analysts.to_csv("../03_data/analysts.csv")


# %%
# CRSP
Ts = pd.date_range("1996-02-01", end="2020-12-01", freq="M")
T = len(Ts)  # months
cols = [
    "close",
    "askhi",
    "bidlo",
    "ask",
    "bid",
    "vol",
    "ret",
    "permco",
    "ncusip",
    "cusip",
    "comnam",
    "tsymbol",
    "ticker",
    "naics",
    "siccd",
    "shrcd",
    "shrcls",
    "exchcd",
    "primexch",
    "namedt",
    "nameendt",
    "shrout",
    "shrsdt",
    "shrenddt",
    "facpr",
    "facshr",
    "divamt",
    "distcd",
    "dlstcd",
    "dlamt",
    "dlret",
]
CRSP = pd.DataFrame(
    np.random.randn(T, len(cols)),
    columns=cols,
    index=Ts,
)
CRSP["permno"] = np.random.randint(low=10000, high=50000, size=T)
CRSP.index.name = "date"
CRSP = CRSP.reset_index()
CRSP.to_parquet("../03_data/crsp_prices_daily.pq")


# %%
# LBC
Ts = pd.date_range("1996-02-01", end="2020-12-01", freq="M")
T = len(Ts)  # months
LBC = pd.DataFrame(
    np.random.randn(T),
    columns=["LBC"],
)
LBC["date"] = np.random.randint(low=72000, high=80000, size=T)
LBC = LBC.set_index("date")
LBC = LBC.sort_index()
LBC.to_csv("../03_data/lbc_monthly.csv")


# %%
# OPT FACTORS
Ts = pd.date_range("1996-02-01", end="2020-12-01", freq="M")
T = len(Ts)  # months
opt_factors = pd.DataFrame(
    np.random.randn(T, 3),
    columns=["IV_RV", "VS", "DCIV_DPIV"],
    index=Ts,
)
opt_factors.index.name = "date"
opt_factors.to_excel("../03_data/OPT Factors.xlsx")


# %%
# Insti share
Ts = pd.date_range("1996-02-01", end="2020-12-01", freq="M")
T = len(Ts)  # months
insti = pd.DataFrame(
    np.random.randn(T),
    columns=["share"],
    index=Ts,
)
insti["permno"] = np.random.randint(low=10000, high=50000, size=T)
insti.index.name = "date"
insti = insti.reset_index()
insti.to_parquet("../03_data/institutional_holdings_13f.pq")


# %%
# TAQ data
Ts = pd.date_range("2020-01-01", end="2020-12-01", freq="5T")
T = len(Ts)  # months
TAQ = pd.DataFrame(
    np.random.randn(T, 2),
    columns=["r2_to_close", "r2_from_prev_close"],
    index=Ts,
)
TAQ["permno"] = np.random.randint(low=10000, high=50000, size=T)
TAQ.index.name = "date"
TAQ = TAQ.reset_index()
TAQ.to_parquet("../03_data/taq/taq_sample.pq")


# %%
