from utils.state import State
from utils.rlmodelinputs import RLModelInputs
from utils.ormodelinputs import ORModelInputs
from typing import List

class JSPEnv():
    def __init__(self, init_mi: RLModelInputs):
        self.state: State
        self.mi: RLModelInputs

        
        self.mi = init_mi
        self.state = State(self.mi)

    # def update_inv(self):
    #     for i in self.mi.I:
    #         inv = self.mi.S_I[i]
    #         for p in self.mi.P:
    #             inv = max(0,
    #                        inv + self.mi.A_star[i] * self.state.schedule[i, p] - self.mi.D_star[i, p])
    #             # if i == 0:
    #             #     print(inv, self.mi.A_star[i], self.mi.D_star[i, p])
    #         self.state.inv[i] = inv
            

    def reset(self):
        # schedule        
        self.state.schedule = dict()
        for i in self.mi.I:
            for p in self.mi.F + [self.mi.P[0]-1]:
                self.state.schedule[i, p] = self.mi.X[i, p]
            for p in self.mi.P[self.mi.schedule_fixed_periods:]:
                self.state.schedule[i, p] = 0

        # inventory
        self.state.inv = dict()
        for i in self.mi.I:
            self.state.inv[i, self.mi.P[0]-1] = self.mi.S_I[i]
            inv = self.mi.S_I[i]
            for p in self.mi.P:
                inv = max(0,
                          inv + self.mi.A_star[i] * self.state.schedule[i, p] - self.mi.D_star[i, p])
                self.state.inv[i, p] = inv

        # stockout
        self.state.stockout = {
            (i, p): max(0, 
                        self.mi.D_star[i, p] - 
                        self.state.inv[i, p] - 
                        self.mi.A_star[i] * self.state.schedule[i, p])            
            for i in self.mi.I
            for p in self.mi.P
        }

        # forcasted demand
        # TODO: 應該只揭露尚未決定 action 的資訊
        self.state.forcasted_demand = {
            (i, p): self.mi.D_star[i, p]
            for i in self.mi.I
            for p in self.mi.P
        }

        self.state.t = 0

        self.obj = self.obj_calculator()


        # self.state.display()
        return self.state.to_numpy()

    def display(self):
        self.state.display()
        print(f"obj: {self.obj}")
        print()

    def step(self, action):
        # if (self.mi.schedule_fixed_periods + self.state.t) >= self.mi.plan_horizon:
        #     raise ValueError("step out of range")


        # udpate schedule
        for i in self.mi.I:
            self.state.schedule[i, self.mi.current_period + self.mi.schedule_fixed_periods + self.state.t] = action[i]

        # update invenrory
        for i in self.mi.I:
            self.state.inv[i, self.mi.P[0]-1] = self.mi.S_I[i]
            inv = self.mi.S_I[i]
            for p in self.mi.P:
                inv = max(0,
                          inv + self.mi.A_star[i] * self.state.schedule[i, p] - self.mi.D_star[i, p])
                self.state.inv[i, p] = inv

        # update stockout
        self.state.stockout = {
            (i, p): max(0, 
                        self.mi.D_star[i, p] - 
                        self.state.inv[i, p] - 
                        self.mi.A_star[i] * self.state.schedule[i, p])            
            for i in self.mi.I
            for p in self.mi.P
        }

        self.state.t += 1

        # self.state.display()
        new_obj = self.obj_calculator()
        reward = new_obj - self.obj
        self.obj = new_obj
        # print(f"revenue: {revenue}")
        # print(f"stockout_cost: {stockout_cost}")
        # print(f"inventory_cost: {inventory_cost}")
        # print(f"transition_cost: {transition_cost}")
        # print(f"reward: {reward}")

        done = (self.mi.schedule_fixed_periods + self.state.t) >= self.mi.plan_horizon
        # print(self.state.to_numpy())
        return self.state.to_numpy(), reward, done, ""

    def obj_calculator(self):
        # revenue
        revenue = 0
        for i in self.mi.I:
            for p in self.mi.P:
                revenue += self.mi.V[i] * (self.mi.D_star[i, p] - self.state.stockout[i, p])

        # stockout cost
        stockout_cost = 0
        for i in self.mi.I:
            for p in self.mi.P:
                stockout_cost += self.state.stockout[i, p] * self.mi.C_L[i]
        
        # inventory cost
        inventory_cost = 0
        for i in self.mi.I:
            for p in self.mi.P:
                inventory_cost += self.state.inv[i, p] * self.mi.C_S[i]
        
        # transition cost
        transition_cost = 0
        for i in self.mi.I:
            if self.mi.X[i, self.mi.P[0]-1] == 1:
                cur = i
                break
        for p in self.mi.P:
            for i in self.mi.I:
                if self.state.schedule[i, p] == 1 and cur != i:
                    transition_cost += self.mi.C_T[cur, i]
                    cur = i

        return revenue - stockout_cost - inventory_cost - transition_cost

    # def step(self, action): # action is the one-hot code of producing item, only one of the action[i] == 1
    #     # stockout cost
    #     stockout = [max(0, 
    #                     self.mi.D_hat[i, self.state.t] - 
    #                     self.state.inv[i] - 
    #                     self.mi.A_hat[i, self.state.t] * action[i]) 
    #                 for i in self.mi.I] # max(0, demand - inventory - production)
    #     stockout_cost = sum([stockout[i] * self.mi.C_L[i] for i in self.mi.I])

    #     sales = [self.mi.D_hat[i, self.state.t] - stockout[i] for i in self.mi.I]
    #     revenue = sum([sales[i] * self.mi.V[i] for i in self.mi.I])


    #     # update state
    #     self.state.inv = [max(0, 
    #                           self.state.inv[i] + 
    #                           self.mi.A_hat[i, self.state.t] * action[i] - 
    #                           self.mi.D_hat[i, self.state.t])
    #                       for i in self.mi.I]
        
    #     for i in self.mi.I:
    #         for t in range(self.mi.schedule_fixed_periods - 1): # fixed schedule shift
    #             self.state.fixed_schedule[i, t] = self.state.fixed_schedule[i, t + 1]
    #         self.state.fixed_schedule[i, self.mi.schedule_fixed_periods - 1] = action[i]
        
    #     for i in self.mi.I:
    #         for t in range(self.mi.plan_horizon):
    #             self.state.forcasted_demand[i, t] = self.nmis[self.state.t].D_star[i, self.state.t + t]

    #     inv_cost = sum([self.state.inv[i] * self.mi.C_S[i] for i in self.mi.I])

    #     self.state.t += 1

    #     reward = revenue - stockout_cost - inv_cost - trans_cost

    #     return self.state, reward, done, info
            