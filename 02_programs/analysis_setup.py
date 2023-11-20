# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by 

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
import os
from joblib import load

import matplotlib
import seaborn as sns

from init_model_parameters import possible_samples


# %% Hyperparameters

# ---- figure stuff:
cm = 1 / 2.54
width = 15.92 * cm
height = 10 * cm

matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.xmargin"] = 0.02
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.edgecolor"] = "0.15"
matplotlib.rcParams["axes.linewidth"] = 1.25
matplotlib.rcParams["figure.dpi"] = 800
matplotlib.rcParams["savefig.dpi"] = 800
matplotlib.rcParams["savefig.transparent"] = True
matplotlib.rcParams["savefig.format"] = "pdf"
matplotlib.rcParams["mathtext.fontset"] = "stixsans"

n_port = 10
n_top_features = 10
n_bot_features = 5


group_dict = {
    "Accruals": "Acc",
    "Profitability": "Prof",
    "Quality": "Q",
    "Investment": "Inv",
    "Industry": "Ind",
    "Illiquidity": "Ill",
    "Informed Trading": "Info",
    "Value": "Val",
    "Past Prices": "Past",
    "Frictions": "Fric",
    "Contract": "C",
    "Risk": "Risk",
}


def sns_palette(n_colors):
    return sns.cubehelix_palette(n_colors=n_colors, start=0, rot=-0.05, dark=0.1, light=0.75)


# %% Load data
print("Possible samples:")
for i, elem in enumerate(possible_samples):
    print("%d: %s" % (i, elem))
# fileLoc = input("Which file location?   ")
fileLoc = 1
fileLoc = possible_samples[int(fileLoc)]
analysisLoc = os.path.join(fileLoc, "analysis")
modelLoc = fileLoc.replace("04_results", "05_models")

prediction_dict = load(os.path.join(modelLoc, "prediction_dict_comparison.pkl"))
# prediction_dict = pd.read_pickle(os.path.join(modelLoc, "prediction_dict_comparison.pkl"))


# %%
