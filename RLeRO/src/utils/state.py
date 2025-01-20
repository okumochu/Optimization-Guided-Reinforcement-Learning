from typing import Dict, List, Tuple
from utils.rlmodelinputs import RLModelInputs
import numpy as np


class State:
    def __init__(self, mi: RLModelInputs) -> None:
        # metadata
        self.mi = mi

        # body
        # inventory levels (product, time)  -> quantity
        self.inv: Dict[Tuple[int, int], bool]
        # ont-hot schedules (product, time) -> T/F
        self.schedule: Dict[Tuple[int, int], bool]
        # forcasted demand (product, time) -> quantity
        self.forcasted_demand: Dict[Tuple[int, int], float]
        self.t: int
        # stockout (product, time) -> quantity
        self.stockout: Dict[Tuple[int, int], float]

    def to_numpy(self):
        # info.
        C_T = np.array([self.mi.C_T[i, j] for i in self.mi.I for j in self.mi.I])
        C_S = np.array([self.mi.C_S[i] for i in self.mi.I])
        C_L = np.array([self.mi.C_L[i] for i in self.mi.I])
        V = np.array([self.mi.V[i] for i in self.mi.I])
        C_T = (C_T - np.min(C_T)) / \
               (np.max(C_T) - np.min(C_T) + 0.001)
        C_S = (C_S - np.min(C_S)) / \
               (np.max(C_S) - np.min(C_S) + 0.001)
        C_L = (C_L - np.min(C_L)) / \
               (np.max(C_L) - np.min(C_L) + 0.001)
        V = (V - np.min(V)) / \
               (np.max(V) - np.min(V) + 0.001)


        padding = self.mi.plan_horizon - len(self.mi.P)
        schedules = list()
        for p in self.mi.P + [self.mi.P[0]-1]:
            d = np.array([self.schedule[i, p] for i in self.mi.I])
            schedules.append(d)
        for p in range(padding):
            d = np.array([0 for i in self.mi.I])
            schedules.append(d)
        schedules = np.concatenate(schedules)

        invs = list()
        for p in self.mi.P:
            d = np.array([self.inv[i, p] for i in self.mi.I])
            invs.append(d)
        for p in range(padding):
            d = np.array([0 for i in self.mi.I])
            invs.append(d)
        invs = np.concatenate(invs)
        invs = (invs - np.min(invs)) / \
               (np.max(invs) - np.min(invs) + 0.001)

        stockouts = list()
        for p in self.mi.P:
            d = np.array([self.stockout[i, p] for i in self.mi.I])
            stockouts.append(d)
        for p in range(padding):
            d = np.array([0 for i in self.mi.I])
            stockouts.append(d)
        stockouts = np.concatenate(stockouts)
        stockouts = (stockouts - np.min(stockouts)) / \
                    (np.max(stockouts) - np.min(stockouts) + 0.001)

        # x = np.concatenate([schedules])
        x = np.concatenate([C_T, C_S, C_L, V, schedules, stockouts, invs, np.array([self.t])])
        # print(x.shape)
        return x

    def display(self) -> None:
        print(f"t = {self.t}")

        print("schedule:")
        for i in self.mi.I:
            for p in self.mi.P:
                print(self.schedule[i, p], end=' ')
                if p == self.mi.current_period + self.mi.schedule_fixed_periods - 1:
                    print("|", end=' ')
            print()

        print("inventory:")
        for i in self.mi.I:
            for p in self.mi.P:
                print(round(self.inv[i, p]), end=' ')
                if p == self.mi.current_period + self.mi.schedule_fixed_periods - 1:
                    print("|", end=' ')
            print()

        print("forcasted demand:")
        for i in self.mi.I:
            for p in self.mi.P:
                print(round(self.forcasted_demand[i, p]), end=' ')
                if p == self.mi.current_period + self.mi.schedule_fixed_periods - 1:
                    print("|", end=' ')
            print()

        print("stockout:")
        for i in self.mi.I:
            for p in self.mi.P:
                print(round(self.stockout[i, p]), end=' ')
                if p == self.mi.current_period + self.mi.schedule_fixed_periods - 1:
                    print("|", end=' ')
            print()
