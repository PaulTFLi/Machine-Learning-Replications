# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

import pandas as pd


def get_python_date_from_matlab_datenum(matlab_datenum):
    from datetime import timedelta
    from datetime import datetime

    python_date = matlab_datenum - 366  # (different zeroing of matlab and python)
    python_date = [datetime.fromordinal(x.astype(int)) + timedelta(days=x % 1) for x in python_date]
    python_date = pd.DatetimeIndex(python_date).round("T")

    return {"Python Date": python_date, "Matlab Datenum": matlab_datenum}


def get_matlab_datenum_from_python_date(python_date):
    from datetime import date

    matlab_datenum = [date.toordinal(d) + d.hour / 24 + d.minute / 24 / 60 + 366 for d in python_date]

    return {"Matlab Datenum": matlab_datenum, "Python Date": python_date}
