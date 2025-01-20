# from configs import *
import pandas as pd
from random import randint, uniform
import os
import matplotlib.pyplot as plt
from utils.ormodelinputs import ORModelInputs
from utils.evaluatorinputs import EvaluatorInput
from utils.rlmodelinputs import RLModelInputs
from utils.state import State

# from robustoptimization.components.uncertainparameter import UncertainParameter
# from robustoptimization.components.uncertaintyset.box import Box


import numpy as np


class DataHandler:
    def __init__(self):
        # configs
        self.current_period = 0
        self.plan_horizon = 15
        self.schedule_fixed_periods = 7
        self.product_cnt = 4
        self.experiment_len = 90  # 30
        self.delta = 1

        self.V_range = (25, 35)
        self.D_hat_range = (80, 120)
        self.A_hat_range_coef = 0.06
        self.A_hat_range = (
            int(
                np.mean(self.D_hat_range)
                * self.product_cnt
                * 0.7
                * (1 + self.A_hat_range_coef)
            ),
            int(
                np.mean(self.D_hat_range)
                * self.product_cnt
                * 0.8
                * (1 - self.A_hat_range_coef)
            ),
        )

        self.I = list(range(self.product_cnt))
        self.P_hat = list(range(self.experiment_len))
        self.__generate_shared_parameters()

    def __generate_shared_parameters(self):
        """Cross decision periods parameters will be initialized here and won't modified thorough experiments"""
        self.D_hat = {
            (i, p): randint(self.D_hat_range[0], self.D_hat_range[1])
            for i in self.I
            for p in self.P_hat
        }
        self.A_hat = {
            (i, p): randint(self.A_hat_range[0], self.A_hat_range[1])
            for i in self.I
            for p in self.P_hat
        }

        self.V = {i: randint(self.V_range[0], self.V_range[1]) for i in self.I}
        self.C_T = {
            (i, j): self.V[j] * np.mean(self.A_hat_range) * 0.1 if i != j else 0
            for i in self.I
            for j in self.I
        }
        self.C_S = {i: self.V[i] * 0.05 for i in self.I}
        self.C_L = {i: self.V[i] * 0.3 for i in self.I}

    def generate_normal_model_inputs(self, last_sols):
        mi = ORModelInputs()

        # Sets
        mi.I = self.I
        mi.P = self.P_hat[self.current_period : self.current_period + self.plan_horizon]
        mi.F = self.P_hat[
            self.current_period : self.current_period + self.schedule_fixed_periods
        ]

        # Normal params
        mi.C_T = self.C_T
        mi.C_S = self.C_S
        mi.C_L = self.C_L
        mi.V = self.V
        # Legacy params
        if self.current_period == 0:
            mi.S_I = {i: 200 for i in self.I}  # TODO: TBD
            mi.X = {
                (i, f): (1 if i == f % len(mi.I) else 0)
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }  # TODO: TBD
        else:
            # last_period = max(last_sols["s"].keys(), key=lambda k: k[1])[1]
            mi.S_I = {i: last_sols["s"][i, mi.P[0] - 1] for i in self.I}
            mi.X = {
                (i, f): round(last_sols["x"][i, f])
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }
        # Uncertain params
        mi.D_star = dict()
        for i in self.I:
            for t, p in enumerate(mi.P):
                mu = self.D_hat[i, p]
                pert = t * self.delta / len(mi.P)
                mi.D_star[i, p] = uniform(mu * (1 - pert), mu * (1 + pert))
        mi.A_star = {i: (self.A_hat_range[0] + self.A_hat_range[1]) / 2 for i in self.I}
        self.plot_demand(mi)

        self.current_period += 1
        return mi

    def generate_robust_model_input(
        self, last_sols, normal_input: ORModelInputs, robustness: float = 0.05
    ):
        mi = ORModelInputs()

        # Sets
        mi.I = self.I
        mi.P = self.P_hat[self.current_period : self.current_period + self.plan_horizon]
        mi.F = self.P_hat[
            self.current_period : self.current_period + self.schedule_fixed_periods
        ]

        # Normal params
        mi.C_T = self.C_T
        mi.C_S = self.C_S
        mi.C_L = self.C_L
        mi.V = self.V
        # Legacy params
        if self.current_period == 0:
            mi.S_I = {i: 200 for i in self.I}  # TODO: TBD
            mi.X = {
                (i, f): (1 if i == f % len(mi.I) else 0)
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }  # TODO: TBD
        else:
            # last_period = max(last_sols["s"].keys(), key=lambda k: k[1])[1]
            mi.S_I = {i: last_sols["s"][i, mi.P[0] - 1] for i in self.I}
            mi.X = {
                (i, f): round(last_sols["x"][i, f])
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }
        # Uncertain params
        mi.D_star = dict()
        for i in self.I:
            for t, p in enumerate(mi.P):
                # mu = self.D_hat[i, p]
                # pert = t*1.5 / len(mi.P)
                # mi.D_star[i, p] = mu * (1 + pert * 0.05)  # robustness
                mi.D_star[i, p] = normal_input.D_star[i, p] * (1 + robustness)
        mi.A_star = {i: self.A_hat_range[0] for i in self.I}
        # mi.A_star = {
        #     i: (self.A_hat_range[0] + self.A_hat_range[1]) / 2 * (1 + pert * 0)
        #     for i in self.I
        # }

        self.plot_demand(mi)
        self.current_period += 1
        return mi

    def generate_evaluator_input(self, xs):
        ei = EvaluatorInput()

        # Sets
        ei.I = self.I
        ei.P_hat = self.P_hat

        # Normal params
        ei.C_T = self.C_T
        ei.C_S = self.C_S
        ei.C_L = self.C_L
        ei.V = self.V

        # Legacy params
        ei.S_I = {i: 200 for i in self.I}  # TODO: TBD
        x_hat = dict()
        for x in xs:
            x_hat.update(x)

        for p in self.P_hat:
            assert sum([x_hat[i, p] for i in self.I]) == 1
        #     for i in self.I:
        #         if x_hat[i, p] == 1:
        #             print(i)

        ei.X = x_hat

        # Uncertain params (realized)
        ei.D_hat = self.D_hat
        ei.A_hat = self.A_hat

        return ei

    def generate_robust_model_input_bk(self, last_sols, robustness):
        raise NotImplementedError("just for backup")
        mi = ORModelInputs()

        # Sets
        mi.I = self.I
        mi.P = self.P_hat[self.current_period : self.current_period + self.plan_horizon]
        mi.F = self.P_hat[
            self.current_period : self.current_period + self.schedule_fixed_periods
        ]

        # Normal params
        mi.C_T = self.C_T
        mi.C_S = self.C_S
        mi.C_L = self.C_L
        mi.V = self.V
        # Legacy params
        if self.current_period == 0:
            mi.S_I = {i: 200 for i in self.I}  # TODO: TBD
            mi.X = {
                (i, f): (1 if i == f % len(mi.I) else 0)
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }  # TODO: TBD
        else:
            # last_period = max(last_sols["s"].keys(), key=lambda k: k[1])[1]
            mi.S_I = {i: last_sols["s"][i, mi.P[0] - 1] for i in self.I}
            mi.X = {
                (i, f): last_sols["x"][i, f] for i in mi.I for f in mi.F + [mi.P[0] - 1]
            }
        # Uncertain params
        mi.A_star = {i: UncertainParameter(f"A_star_{i}") for i in self.I}
        mi.D_star = {
            (i, p): UncertainParameter(f"D_star_{i}_{p}") for i in self.I for p in mi.P
        }

        # Uncertainty sets
        """
        for i in self.I:
            for t, p in enumerate(mi.P):
                mu = self.D_hat[i, p]
                pert = t*0.5 / len(mi.P)
                mi.D_star[i, p] = uniform(mu*(1-pert), mu*(1+pert))
        """
        D_star_nominals = dict()
        D_star_shifts = dict()
        for i in self.I:
            for t, p in enumerate(mi.P):
                mu = self.D_hat[i, p]
                pert = t * 0.5 / len(mi.P)
                D_star_nominals[i, p] = uniform(mu * (1 - pert), mu * (1 + pert))
                D_star_shifts[i, p] = mu * pert

        balance_nominals = dict()
        balance_shifts = dict()
        for i in self.I:
            for t, p in enumerate(mi.P):
                balance_nominals[i, p] = {
                    mi.A_star[i]: (self.A_hat_range[0] + self.A_hat_range[1]) / 2,
                    mi.D_star[i, p]: D_star_nominals[i, p],
                }

                balance_shifts[i, p] = [
                    {
                        mi.A_star[i]: (self.A_hat_range[1] - self.A_hat_range[0]) / 2,
                        mi.D_star[i, p]: D_star_shifts[i, p],
                    }
                ]
        mi.balance_unc_sets = {
            (i, p): Box(
                f"balance_box_{i}_{p}",
                robustness=robustness,
                nominal_data=balance_nominals[i, p],
                base_shifts=balance_shifts[i, p],
            )
            for i in self.I
            for p in mi.P
        }

        objective_nominals = dict()
        objective_shifts = [dict()]
        for i in self.I:
            for t, p in enumerate(mi.P):
                objective_nominals[mi.D_star[i, p]] = D_star_nominals[i, p]
                objective_shifts[0][mi.D_star[i, p]] = D_star_shifts[i, p]
        mi.objective_unc_set = Box(
            "obj_box",
            robustness=robustness,
            nominal_data=objective_nominals,
            base_shifts=objective_shifts,
        )
        return mi

    def generate_RL_model_input(
        self, last_sols, normal_input: ORModelInputs, inference: bool = False
    ) -> RLModelInputs:
        # just for initialization
        mi = RLModelInputs()
        mi.schedule_fixed_periods = self.schedule_fixed_periods
        mi.plan_horizon = self.plan_horizon
        mi.current_period = self.current_period

        mi.I = self.I
        # mi.P = self.P_hat[:self.plan_horizon]
        # mi.F = self.P_hat[:self.schedule_fixed_periods]
        mi.P = self.P_hat[self.current_period : self.current_period + self.plan_horizon]
        mi.F = self.P_hat[
            self.current_period : self.current_period + self.schedule_fixed_periods
        ]

        # Normal params
        mi.V = self.V
        mi.C_L = self.C_L
        mi.C_S = self.C_S
        mi.C_T = self.C_T

        # Legacy params
        if self.current_period == 0:
            mi.S_I = {i: 200 for i in self.I}  # TODO: TBD
            mi.X = {
                (i, f): (1 if i == f % len(mi.I) else 0)
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }  # TODO: TBD
        else:
            # last_period = max(last_sols["s"].keys(), key=lambda k: k[1])[1]
            mi.S_I = {i: last_sols["s"][i, mi.P[0] - 1] for i in self.I}
            mi.X = {
                (i, f): round(last_sols["x"][i, f])
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }

        if inference:
            # Uncertain params
            mi.D_star = dict()
            for i in self.I:
                for t, p in enumerate(mi.P):
                    mi.D_star[i, p] = normal_input.D_star[i, p]
            mi.A_star = {i: self.A_hat_range[0] for i in self.I}
        else:  # training
            mi.D_star = dict()
            for i in self.I:
                for t, p in enumerate(mi.P):
                    mu = self.D_hat[i, p]
                    pert = t * self.delta / len(mi.P)
                    mi.D_star[i, p] = uniform(mu * (1 - pert), mu * (1 + pert))
            mi.A_star = {
                i: (self.A_hat_range[0] + self.A_hat_range[1]) / 2 for i in self.I
            }

        self.current_period += 1
        return mi

    def generate_DM_oracle_input(self, state: State):
        mi = ORModelInputs()

        current_period = self.current_period - 1

        # Sets
        mi.I = self.I
        mi.P = self.P_hat[current_period : current_period + self.plan_horizon]
        mi.F = self.P_hat[current_period : current_period + self.schedule_fixed_periods]

        # Normal params
        mi.C_T = self.C_T
        mi.C_S = self.C_S
        mi.C_L = self.C_L
        mi.V = self.V

        # Legacy params
        if current_period == 0:
            mi.S_I = {i: 200 for i in self.I}  # TODO: TBD
            mi.X = {
                (i, f): (1 if i == f % len(mi.I) else 0)
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }  # TODO: TBD
        else:
            # last_period = max(last_sols["s"].keys(), key=lambda k: k[1])[1]
            mi.S_I = {i: state.inv[i, mi.P[0] - 1] for i in self.I}
            mi.X = {
                (i, f): round(state.schedule[i, f])
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }
        # Uncertain params
        mi.D_star = dict()
        for i in self.I:
            for t, p in enumerate(mi.P):
                mu = self.D_hat[i, p]
                pert = t * self.delta / len(mi.P)
                mi.D_star[i, p] = uniform(mu * (1 - pert), mu * (1 + pert))
        mi.A_star = {i: (self.A_hat_range[0] + self.A_hat_range[1]) / 2 for i in self.I}
        return mi

    def generate_RM_oracle_input(self, state: State, robustness):
        mi = ORModelInputs()

        current_period = self.current_period - 1

        # Sets
        mi.I = self.I
        mi.P = self.P_hat[current_period : current_period + self.plan_horizon]
        mi.F = self.P_hat[current_period : current_period + self.schedule_fixed_periods]

        # Normal params
        mi.C_T = self.C_T
        mi.C_S = self.C_S
        mi.C_L = self.C_L
        mi.V = self.V

        # Legacy params
        if current_period == 0:
            mi.S_I = {i: 200 for i in self.I}  # TODO: TBD
            mi.X = {
                (i, f): (1 if i == f % len(mi.I) else 0)
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }  # TODO: TBD
        else:
            # last_period = max(last_sols["s"].keys(), key=lambda k: k[1])[1]
            mi.S_I = {i: state.inv[i, mi.P[0] - 1] for i in self.I}
            mi.X = {
                (i, f): round(state.schedule[i, f])
                for i in mi.I
                for f in mi.F + [mi.P[0] - 1]
            }
        # Uncertain params
        mi.D_star = dict()
        for i in self.I:
            for t, p in enumerate(mi.P):
                mu = self.D_hat[i, p]
                pert = t * 1.5 / len(mi.P)
                mi.D_star[i, p] = uniform(mu * (1 - pert), mu * (1 + pert)) * (
                    1 + robustness
                )
        mi.A_star = {i: self.A_hat_range[0] for i in self.I}
        return mi

    def export(self, name, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        with pd.ExcelWriter(
            os.path.join(save_dir, f"{name}.xlsx"), engine="openpyxl"
        ) as writer:
            # C_T
            C_T = pd.DataFrame(
                {f"j={j}": [self.C_T[i, j] for i in self.I] for j in self.I},
                index=[f"i={i}" for i in self.I],
            )
            C_T.to_excel(writer, sheet_name="C_T")

            # C_T
            C_S = pd.DataFrame({f"i={i}": [self.C_S[i]] for i in self.I})
            C_S.to_excel(writer, sheet_name="C_S")

            # C_L
            C_L = pd.DataFrame({f"i={i}": [self.C_L[i]] for i in self.I})
            C_L.to_excel(writer, sheet_name="C_L")

            # V
            V = pd.DataFrame({f"i={i}": [self.V[i]] for i in self.I})
            V.to_excel(writer, sheet_name="V")

            # S_I
            S_I = {i: 200 for i in self.I}
            S_I = pd.DataFrame({f"i={i}": [S_I[i]] for i in self.I})
            S_I.to_excel(writer, sheet_name="S_I")

            # X
            F = self.P_hat[0 : 0 + self.schedule_fixed_periods]
            X = {
                (i, f): (1 if i == f % len(self.I) else 0)
                for i in self.I
                for f in F + [F[0] - 1]
            }
            X = pd.DataFrame(
                {f"p={f}": [X[i, f] for i in self.I] for f in sorted(F + [F[0] - 1])},
                index=[f"i={i}" for i in self.I],
            )
            X.to_excel(writer, sheet_name="X")

            # D_hat
            D_hat = pd.DataFrame(
                {f"p={p}": [self.D_hat[i, p] for i in self.I] for p in self.P_hat},
                index=[f"i={i}" for i in self.I],
            )
            D_hat.to_excel(writer, sheet_name="D_hat")

            # A_hat
            A_hat = pd.DataFrame(
                {f"p={p}": [self.A_hat[i, p] for i in self.I] for p in self.P_hat},
                index=[f"i={i}" for i in self.I],
            )
            A_hat.to_excel(writer, sheet_name="A_hat")

    def plot_demand(self, mi):
        os.makedirs("outputs", exist_ok=True)

        # fig, ax = plt.subplots(self.product_cnt, 1, figsize=(
        #     20, 12), constrained_layout=True)
        # fig.suptitle("Demand for each products", fontsize=16)

        # for i in self.I:
        #     actual = [self.D_hat[i, p] for p in mi.P]
        #     predict = [mi.D_star[i, p] for p in mi.P]
        #     error = [abs(predict[p] - actual[p])
        #                              for p in range(len(mi.P))]

        #     ax[i].set_title(f'Product {i}')
        #     ax[i].plot(actual, label="Actual")
        #     ax[i].plot(predict, label="Expected")
        #     ax[i].plot(error, label="Error")
        #     ax[i].legend()

        plot_actual = [self.D_hat[0, p] for p in mi.P]
        plot_predict = [mi.D_star[0, p] for p in mi.P]
        plot_residual = [
            abs(plot_predict[p] - plot_actual[p]) for p in range(len(mi.P))
        ]

        plt.title("Demand simulation")
        plt.plot(mi.P, plot_actual, label="actual", linestyle="-")
        plt.plot(mi.P, plot_predict, label="estimation", linestyle="--")
        plt.plot(mi.P, plot_residual, label="error", linestyle=":")
        plt.xlabel("period")
        plt.ylabel("quantity")
        plt.legend()
        plt.savefig(os.path.join("outputs", f"demand-{self.current_period}.jpg"))
        plt.close()
