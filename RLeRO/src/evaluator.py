from utils.evaluatorinputs import EvaluatorInput
import gurobipy as gp
from gurobipy import GRB
import os
import datetime
import plotly.express as px


class Evaluator:
    def __init__(self, evaluator_inputs: EvaluatorInput) -> None:
        self.model = gp.Model()
        self.ei = evaluator_inputs

        self.add_vars()
        self.add_objective()
        self.add_constraints()

    def add_vars(self):
        self.x = self.model.addVars(
            self.ei.I, self.ei.P_hat + [self.ei.P_hat[0]-1],
            vtype=GRB.BINARY,
            name="x"
        )
        self.s = self.model.addVars(
            self.ei.I, self.ei.P_hat + [self.ei.P_hat[0]-1],
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="s"
        )
        self.l = self.model.addVars(
            self.ei.I, self.ei.P_hat,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="l"
        )
        self.z = self.model.addVars(
            self.ei.I, self.ei.I, self.ei.P_hat,
            vtype=GRB.BINARY,
            name="z"
        )

    def add_objective(self):
        self.sales_profit = self.model.addVar(
            vtype=GRB.CONTINUOUS, lb=0.0, name="sales_profit")
        self.inventory_cost = self.model.addVar(
            vtype=GRB.CONTINUOUS, lb=0.0, name="inventory_cost")
        self.stockout_cost = self.model.addVar(
            vtype=GRB.CONTINUOUS, lb=0.0, name="stockout_cost")
        self.transition_cost = self.model.addVar(
            vtype=GRB.CONTINUOUS, lb=0.0, name="transition_cost")

        self.model.addConstr(
            self.sales_profit == gp.quicksum(
                self.ei.V[i] * (self.ei.D_hat[i, p] - self.l[i, p]) for i in self.ei.I for p in self.ei.P_hat)
        )
        self.model.addConstr(
            self.inventory_cost == gp.quicksum(
                self.ei.C_S[i] * self.s[i, p] for i in self.ei.I for p in self.ei.P_hat)
        )
        self.model.addConstr(
            self.stockout_cost == gp.quicksum(
                self.ei.C_L[i] * self.l[i, p] for i in self.ei.I for p in self.ei.P_hat)
        )
        self.model.addConstr(
            self.transition_cost == gp.quicksum(self.ei.C_T[i, j] * self.z[i, j, p]
                                                for i in self.ei.I for j in self.ei.I for p in self.ei.P_hat if i != j)
        )

        self.model.setObjective(
            self.sales_profit - self.inventory_cost -
            self.stockout_cost - self.transition_cost,
            GRB.MAXIMIZE
        )

    def add_constraints(self):
        # 起始狀態定義
        for i in self.ei.I:
            self.model.addConstr(
                self.s[i, self.ei.P_hat[0]-1] == self.ei.S_I[i]
            )

            for p in self.ei.P_hat + [self.ei.P_hat[0]-1]:
                self.model.addConstr(
                    self.x[i, p] == self.ei.X[i, p]
                )

        # 生產、倉儲、需求平衡
        for i in self.ei.I:
            for p in self.ei.P_hat:

                self.model.addConstr(
                    self.s[i, p] == self.s[i, p-1] + self.ei.A_hat[i, p] *
                    self.x[i, p] - self.ei.D_hat[i, p] + self.l[i, p]
                )

        # 產品轉換
        for j in self.ei.I:
            for p in self.ei.P_hat:
                self.model.addConstr(
                    gp.quicksum(self.z[i, j, p]
                                for i in self.ei.I) == self.x[j, p]
                )
        for i in self.ei.I:
            for p in self.ei.P_hat:  # for p in self.mi.P[1:]:
                self.model.addConstr(
                    gp.quicksum(self.z[i, j, p]
                                for j in self.ei.I) == self.x[i, p-1]
                )

        # 生產限制
        for p in self.ei.P_hat:
            self.model.addConstr(
                gp.quicksum(self.x[i, p] for i in self.ei.I) == 1
            )

    def __optimize(self, log=False):
        self.model.setParam("LogToConsole", int(log))
        self.model.optimize()

    def __get_sols(self):
        self.sols = dict()
        # vars
        self.sols["x"] = self.model.getAttr("x", self.x)
        self.sols["s"] = self.model.getAttr("x", self.s)
        self.sols["l"] = self.model.getAttr("x", self.l)
        self.sols["z"] = self.model.getAttr("x", self.z)
        # objs
        self.sols["obj"] = self.model.getObjective().getValue()
        self.sols["sales_profit"] = self.sales_profit.X
        self.sols["inventory_cost"] = self.inventory_cost.X
        self.sols["stockout_cost"] = self.stockout_cost.X
        self.sols["transition_cost"] = self.transition_cost.X
        return self.sols

    def evaluate(self):
        self.__optimize()
        self.sols = self.__get_sols()
        return self.sols

    def generate_gantt(self):
        os.makedirs("outputs", exist_ok=True)
        start_date = datetime.datetime(
            2022, 6, 1)

        schedule = list()
        for p in self.ei.P_hat:
            for i in self.ei.I:
                if self.sols["x"][i, p] == 1:
                    schedule.append(i)
                    break

        df = list()
        product = -1
        start = -1
        finish = -1
        colors = [f"Product {i}" for i in self.ei.I]
        for p, i in enumerate(schedule):
            if product != i:
                start = p
                product = i
            if p+1 != len(schedule) and schedule[p+1] != product:
                finish = p+1
                df.append(
                    dict(Task=f"Product {i}", Start=start, Finish=finish, Color=colors[i]))
        for t in df:
            t["Start"] = start_date + datetime.timedelta(t["Start"])
            t["Finish"] = start_date + datetime.timedelta(t["Finish"])

        discrete_map_resource = {
            'Product 0': 'blue',
            'Product 1': 'red',
            'Product 2': 'purple',
            'Product 3': 'green'}

        fig = px.timeline(df, x_start="Start", x_end="Finish",
                          y="Task", color="Color", color_discrete_map=discrete_map_resource)
        fig.update_yaxes(categoryorder='array', categoryarray=[
                         f"Product {i}" for i in self.ei.I])
        fig.write_html(os.path.join(
            "outputs", f"{start_date}-gantt.html"))
