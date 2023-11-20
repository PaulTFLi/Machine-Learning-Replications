# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
# Packages
import pandas as pd
import os
import glob


# %%
# Files
ensembles = {
    "L_EN": ["Ridge", "ElasticNet", "Lasso", "PCR", "PLS"],
    "N_EN": ["FFNN", "LightGBR", "LightRF", "LightDart"],
}
N_STEPS = 18
modelLoc = "/mnt/wwu1/ec_bronze/_nogroup/d994a2f0-4659-40a1-b872-085bbf230ad8/_projects/h_beck18/options_ml/05_models/option_sample/full"


# %%
# Loop through models and average SHAPs by ensemble
for ensemble, models in ensembles.items():
    print(f"{ensemble}\n")
    for step in range(N_STEPS):
        print(step)
        for i_model, model in enumerate(models):
            print(model)
            loc = glob.glob(os.path.join(modelLoc, f"{model}*___full"))[0]
            shap_file = os.path.join(loc, f"shaps_{step}.pq")
            shaps = pd.read_parquet(shap_file)
            if i_model == 0:
                all_shaps = shaps.copy()
            else:
                all_shaps = all_shaps + shaps
        all_shaps /= len(models)
        all_shaps.to_parquet(
            os.path.join(f"/scratch/tmp/h_beck18/__options_cs_ml/04_results/shaps/{ensemble}_{step}.pq")
        )


# %%
