# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

import numpy as np
import pandas as pd
from pylatex import Tabular, Math


def dollar(x):
    return Math(data=x, inline=True, escape=False)


import collections


class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)


variable_dict = {
    "predicted": dollar(r"\hat{r}"),
    "target": dollar(r"r"),
    "abs_error": dollar(r"|\hat{e}|"),
    "abs_xs_error": dollar(r"|\hat{e}_{XS}|"),
    "retail_share": dollar(r"I^R"),
    "inst_share": dollar(r"I^I"),
    "retail_share:inst_share": dollar(r"I^R \times I^I"),
}


def entity_flag(flag):
    if flag:
        return "Yes"
    else:
        return "-"


def float_to_str(numvar, prec=3):
    if np.isnan(numvar):
        return ""
    else:
        fmt = "{:." + str(prec) + "f}"
        return fmt.format(numvar)


def thousander_separator(value):
    return "{:,}".format(value)


def sig_stars(val):
    return "*" * np.sum(np.abs(val) > np.array([1.6449, 1.96, 2.5758]))


def replacer(string, dct):
    for old, new in dct.items():
        if isinstance(new, str):
            string = string.replace(old, new)
        else:
            if string == old:
                string = new
                break
    return string


def each_specification(res, t=pd.Series()):

    if not (t.empty):
        tstats = t
    else:
        tstats = res.tstats

    params = pd.Series(
        [(float_to_str(x, 3) + sig_stars(y), float_to_str(y)) for x, y in zip(res.params, tstats)],
        index=res.params.index,
    )
    params["Observations"] = thousander_separator(res.nobs)
    params["R-squared"] = float_to_str(res.rsquared)
    params["Entity FE"] = entity_flag(res.model.entity_effects)
    params["Time FE"] = entity_flag(res.model.time_effects)
    params["Other FE"] = entity_flag(res.model.other_effects)
    dependent = pd.Series(res.model.dependent.vars[0], index=["Dependent"])

    return pd.concat([dependent, params], sort=False)


def single_ols(res, factor_intercept):
    param = res.params.copy()
    names = param.index.values
    idx_intercept = np.argwhere(names == "Intercept")
    param[idx_intercept] *= factor_intercept

    param = [float_to_str(x, 2) for x in param.values]
    tval = res.tvalues
    tval_stars = [sig_stars(x) for x in tval]
    tval = [float_to_str(x, 2) for x in tval.values]
    tval = [x + y for x, y in zip(tval, tval_stars)]
    values = list(zip(param, tval))
    r2 = float_to_str(res.rsquared * 100, 2)
    r2adj = float_to_str(res.rsquared_adj * 100, 2)

    dat = pd.Series()
    for name, val in zip(names, values):
        dat[name] = val
    dat["R2"] = r2
    dat["R2 adj"] = r2adj
    dat.name = res.model.endog_names

    return pd.DataFrame(dat).T


def data_block(tabular, data):
    old_index = []
    for index, row in data.iterrows():
        index_tmp = index
        if old_index == index:
            index = ""
        if np.any([isinstance(x, list) for x in row]) | np.any([isinstance(x, tuple) for x in row]):
            cells = [index]
            for r in row:
                if isinstance(r, list) | isinstance(r, tuple):
                    cells.append(r[0])
                else:
                    cells.append(r)
            tabular.add_row(cells)
            cells = [""]
            for r in row:
                if isinstance(r, list) | isinstance(r, tuple):
                    cells.append("(" + r[1] + ")")
                else:
                    cells.append("")
            tabular.add_row(cells)
        else:
            cells = [index]
            for r in row:
                cells.append(r)
            tabular.add_row(cells)
        old_index = index_tmp


def panel_ols_to_table(
    results,
    file,
    kick=None,
    caption=None,
    caption_title=None,
    label_str=None,
    se_cluster=None,
    entity_name=None,
    time_name=None,
    other_name=None,
    column_names=None,
    left_column_title=None,
    additional_rows=pd.Series(),
    t_stats=[],
    t_stat_type="Robust",
):
    all_info = ["Observations", "R-squared", "Entity FE", "Time FE", "Other FE"]
    if len(t_stats) == 0:
        data = pd.concat([each_specification(x) for x in results], axis=1, sort=False)
    else:
        t_stats_sel = [y.loc[t_stat_type] for y in t_stats]
        res_tuple = zip(results, t_stats_sel)
        data = pd.concat([each_specification(x[0], x[1]) for x in res_tuple], axis=1, sort=False)
    data = data.replace(np.nan, "")
    variable = [x for x in data.index.values if x not in all_info]
    variable.extend(all_info)
    data = data.reindex(variable)
    if kick:
        data.drop(kick, axis=0, inplace=True)

    if entity_name:
        if "Entity FE" in data.index:
            if isinstance(entity_name, str):
                data.rename({"Entity FE": entity_name + " FE"})
            else:
                data.loc["Entity FE"] = entity_name
    if time_name:
        if "Time FE" in data.index:
            if isinstance(time_name, str):
                data.rename({"Time FE": time_name + " FE"}, axis=0, inplace=True)
            else:
                data.loc["Time FE"] = time_name

    if other_name:
        if "Other FE" in data.index:
            if isinstance(other_name, str):
                data.rename({"Other FE": other_name + " FE"}, axis=0, inplace=True)
            else:
                data.loc["Other FE"] = other_name

    if not (se_cluster is None):
        data.loc["SEs"] = se_cluster

    if "Dependent" in data.index:
        data.loc["Dependent"] = [replacer(x, variable_dict) for x in data.loc["Dependent"]]

    if not (additional_rows.empty):
        for key, value in additional_rows.items():
            data.loc[key] = value

    data = data.reset_index()
    data["index"] = [replacer(x, variable_dict) for x in data["index"]]

    ncol = data.shape[1]
    tabular = Tabular("l" + "c" * (ncol - 1), booktabs=True)

    if left_column_title:
        cells = [left_column_title]
    else:
        cells = [""]
    if column_names:
        cells.extend(column_names)
    else:
        for i in range(1, ncol):
            cells.append("(" + str(i) + ")")
    tabular.add_row(cells)
    # tabular.add_hline(1, ncol)

    for _, row in data.iterrows():
        index = row.iloc[0]
        row = row.iloc[1:]
        if index == "Observations":
            tabular.add_hline(1, ncol)

        if np.any([isinstance(x, list) for x in row]) | np.any([isinstance(x, tuple) for x in row]):
            cells = [index]
            for r in row:
                if isinstance(r, list) | isinstance(r, tuple):
                    cells.append(r[0])
                else:
                    cells.append(r)
            tabular.add_row(cells)
            cells = [""]
            for r in row:
                if isinstance(r, list) | isinstance(r, tuple):
                    cells.append("(" + r[1] + ")")
                else:
                    cells.append("")
            tabular.add_row(cells)
        else:
            cells = [index]
            for r in row:
                cells.append(r)
            tabular.add_row(cells)

        if index == "Dependent":
            tabular.add_hline()

    tabular.generate_tex(file.replace(".tex", ""))


# %%
