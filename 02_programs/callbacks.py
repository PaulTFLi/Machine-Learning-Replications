# -*- coding: utf-8 -*-
"""
"Option Return Predictability with Machine Learning and Big Data"

by

Turan G. Bali, Heiner Beckmeyer, Mathis Moerke, and Florian Weigert

January 2023
"""

# %%
from ray.tune import Callback
import numpy as np


# %% Callback class:
class MyCallback(Callback):
    def __init__(
        self,
        mode: str,
        exp_logfile: str,
        exp_top_models: int = 10,
        exp_num_results: int = 100,
        exp_grace_period: int = 0,
        exp_tol: float = 0.0,
    ):
        self._exp_logfile = exp_logfile
        self._mode = mode
        self._exp_top_models = exp_top_models
        self._exp_grace_period = exp_grace_period
        self._exp_num_results = exp_num_results
        self._exp_tol = exp_tol

        self.done_trials = []
        self.exp_no_imp = 0
        self.exp_top_mean = 10000000.0
        self.exp_top_values = []
        self.mean_improvements = []
        self.scores = {}
        self.done = 0

    def experiment_logger(self, config, trial_id, best, iterations):
        with open(self._exp_logfile, "a+") as log:
            print("===== Trial with id %s done at epoch %d." % (trial_id, iterations), file=log)
            print(config, file=log)
            print("%d trials done." % len(self.done_trials), file=log)
            print(
                "Trials without improvements: %d/%d." % (self.exp_no_imp, self._exp_num_results),
                file=log,
            )
            print("This trial's best score: %.6f" % best, file=log)
            print("Last %d mean improvements (in percent):" % self._exp_top_models, file=log)
            print(
                "   ".join("%.6f" % i for i in self.mean_improvements[-self._exp_top_models :]),
                file=log,
            )
            print("Mean of top %d trials: %.6f." % (self._exp_top_models, self.exp_top_mean), file=log)
            print("\n\n", file=log)

    def on_trial_complete(self, iteration: int, trials, trial, **info):
        iterations = len(self.scores[trial])
        self.done += 1
        if self._mode == "min":
            best = min(self.scores[trial])
        else:
            best = max(self.scores[trial])
        self.done_trials.append(best)
        if self._mode == "min":
            exp_top_values = sorted(self.done_trials)[: self._exp_top_models]
        else:
            exp_top_values = sorted(self.done_trials)[-self._exp_top_models :]

        # get mean score.
        mean_finished_trials = np.mean(exp_top_values)
        self.mean_improvements.append(
            (self.exp_top_mean - mean_finished_trials) / self.exp_top_mean
        )  # mean improv in %

        # check for mean improvements.
        if self.mean_improvements[-1] <= self._exp_tol:
            self.exp_no_imp += 1
        else:
            self.exp_no_imp = 0
        self.exp_top_mean = mean_finished_trials

        # ---- logging:
        self.experiment_logger(trial.config, trial, best, iterations)

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if trial not in self.scores.keys():
            self.scores[trial] = []
        self.scores[trial].append(result["score"])


# %% trial name creator
def trial_name_creator(trial):
    """
    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    return str(trial) + "_" + str(trial.experiment_tag[:25])


# %%
