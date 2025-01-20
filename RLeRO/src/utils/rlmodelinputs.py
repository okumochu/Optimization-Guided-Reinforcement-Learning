from configs import *
import pandas as pd
import random
import os


class RLModelInputs:
    def __init__(self):
        self.schedule_fixed_periods: int
        self.plan_horizon: int
        self.current_period: int
        self.__sets()
        self.__params()

    def __sets(self):
        self.I = None
        self.P = None
        self.F = None

    def __params(self):
        self.C_T = None
        self.D_star = None
        self.C_S = None
        self.C_L = None
        self.A_star = None
        self.V = None
        self.S_I = None
        self.X = None

        self.A_hat = None
        self.D_hat = None


    def export(self, name, save_dir):
        raise NotImplementedError()
        os.makedirs(save_dir, exist_ok=True)

        with pd.ExcelWriter(os.path.join(save_dir, f'{name}.xlsx'), engine="openpyxl") as writer:
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
            S_I = pd.DataFrame({f"i={i}": [self.S_I[i]] for i in self.I})
            S_I.to_excel(writer, sheet_name="S_I")

            # X
            X = pd.DataFrame(
                {f"p={f}": [self.X[i, f] for i in self.I]
                    for f in sorted(self.F + [self.P[0]-1])},
                index=[f"i={i}" for i in self.I]
            )
            X.to_excel(writer, sheet_name="X")

            # D_star
            D_star = pd.DataFrame(
                {f"p={p}": [self.D_star[i, p] for i in self.I]
                    for p in self.P},
                index=[f"i={i}" for i in self.I]
            )
            D_star.to_excel(writer, sheet_name="D_star")

            # A_star
            A_star = pd.DataFrame({f"i={i}": [self.A_star[i]] for i in self.I})
            A_star.to_excel(writer, sheet_name="A_star")
