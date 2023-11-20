# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

import numpy as np


def insert_resizebox(file):
    f = open(file)
    lines = f.readlines()
    f.close()

    # insert resizebox
    start_resizebox = "\\resizebox{\textwidth}{!}{%\n"
    end_resizebox = "}%\n"

    idx_tabular = np.argwhere(["tabular" in line for line in lines]).T[0]
    i = 0
    for idx in idx_tabular:
        if np.mod(i, 2) == 0:
            text = start_resizebox
        else:
            text = end_resizebox
        lines.insert(idx, text)

    with open(file, mode="w") as f:
        f.write("% generated by python\n")
    f.close()

    with open(file, mode="a") as f:
        for line in lines:
            f.write(line)
    f.close()
