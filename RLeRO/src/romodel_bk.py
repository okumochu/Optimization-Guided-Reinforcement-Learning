from utils.ormodelinputs import ModelInputs

from robustoptimization.robustlinearmodel import RobustLinearModel
from robustoptimization.components.uncertainparameter import UncertainParameter
from robustoptimization.components.uncertaintyset.box import Box
from robustoptimization.utils.constants import *
from robustoptimization.utils.plotter import generate_evaluations_plot
from robustoptimization.utils.metrics import mean_value_of_robustization, improvement_of_std, robust_rate
from datetime import datetime
from typing import Union
from tqdm import tqdm

import os
import random

"""
Consider not to implement
"""


class ROCPSModel:
    def __init__(self, name: str, model_inputs: ModelInputs, log_path: str, figure_dir: str):
        self.start_timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        self.rlm = RobustLinearModel(log_path, name)
        self.figure_dir = figure_dir
        self.mi = model_inputs

    def add_vars(self):
        self.x = {
            (i, p): self.rlm.add_var(f"x_{i}_{p}", type_=BINARY)
            for i in self.mi.I
            for p in self.mi.P + [self.mi.P[0]-1]
        }

        self.s = {
            (i, p): self.rlm.add_var(f"s_{i}_{p}", lb=0.0, type_=CONTINUOUS)
            for i in self.mi.I
            for p in self.mi.P + [self.mi.P[0]-1]
        }

        self.l = {
            (i, p): self.rlm.add_var(f"l_{i}_{p}", lb=0.0, type_=CONTINUOUS)
            for i in self.mi.I
            for p in self.mi.P
        }

        self.z = {
            (i, j, p): self.rlm.add_var(f"z_{i}_{j}_{p}", type_=BINARY)
            for i in self.mi.I
            for j in self.mi.I
            for p in self.mi.P
        }

    def add_objective(self):
        sales_profit = 0
        inventory_cost = 0
        stockout_cost = 0
        transition_cost = 0

        for i in self.mi.I:
            for p in self.mi.P:
                sales_profit += self.mi.V[i] * \
                    (self.mi.D_star[i, p] - self.l[i, p])

        for i in self.mi.I:
            for p in self.mi.P:
                inventory_cost += self.mi.C_S[i] * self.s[i, p]

        for i in self.mi.I:
            for p in self.mi.P:
                stockout_cost += self.mi.C_L[i] * self.l[i, p]

        for i in self.mi.I:
            for j in self.mi.J:
                if i == j:
                    continue
                for p in self.mi.P:
                    transition_cost += self.mi.C_T[i, j] * self.z[i, j, p]

        objective = -(sales_profit - inventory_cost -
                      stockout_cost - transition_cost)

        self.rlm.set_objective(objective, sense=MINIMIZE,
                               uncertainty_set=self.mi.objective_unc_set)

    def add_constraints(self):
        pass
