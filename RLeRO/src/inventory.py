import numpy as np
import gym
from gym import spaces
import cvxpy as cp
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class InventoryEnv(gym.Env):
    """
    A custom environment for inventory management.
    
    State:
      - inventory level (I_t)
      - last observed demand (used as a simple proxy for future demand)
      - lead time (constant in this example)
      - time indicator (normalized time, e.g., seasonality indicator)
    
    Action:
      - Order quantity for the next period (continuous, Box space)
      
    Reward:
      - Negative of the total cost, which includes:
          Holding cost (for excess inventory)
          Shortage cost (for unmet demand)
          Ordering cost (per unit ordered)
          
    Transition:
      I_{t+1} = I_t + Q_t - D_t
    """
    def __init__(self):
        super(InventoryEnv, self).__init__()
        # Parameters
        self.initial_inventory = 50.0
        self.max_inventory = 100.0
        self.C_h = 1.0   # holding cost per unit
        self.C_s = 10.0  # shortage cost per unit
        self.C_o = 2.0   # ordering cost per unit (or per order if fixed cost is added)
        self.lead_time = 1  # for simplicity, lead time is fixed
        self.max_steps = 50  # episode length

        # Action space: order quantity between 0 and 50 units
        self.action_space = spaces.Box(low=0, high=50, shape=(1,), dtype=np.float32)
        # Observation space: [inventory, last demand, lead time, time indicator]
        low_obs = np.array([0, 0, 0, 0], dtype=np.float32)
        high_obs = np.array([self.max_inventory, 100, 10, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.inventory = self.initial_inventory
        self.last_demand = 10.0  # initial dummy value
        self.time_indicator = 0.0  # normalized time [0,1]
        self.state = np.array([self.inventory, self.last_demand, self.lead_time, self.time_indicator], dtype=np.float32)
        return self.state

    def step(self, action):
        # Action: order quantity
        order_qty = float(action[0])
        # Simulate demand: for demonstration, use a Poisson distribution with lambda = last_demand
        demand = np.random.poisson(lam=self.last_demand)
        # Update inventory level: current inventory + order - demand
        new_inventory = self.inventory + order_qty - demand
        
        # Calculate costs:
        holding_cost = self.C_h * max(0, new_inventory)       # cost for excess inventory
        shortage_cost = self.C_s * max(0, -new_inventory)       # penalty for stockouts
        ordering_cost = self.C_o * order_qty                  # ordering cost
        total_cost = holding_cost + shortage_cost + ordering_cost
        
        # Reward is negative total cost
        reward = - total_cost
        
        # Update state: assume inventory cannot be negative (shortage is penalized)
        self.inventory = max(0, new_inventory)
        self.last_demand = demand
        self.current_step += 1
        self.time_indicator = self.current_step / self.max_steps  # normalized time
        self.state = np.array([self.inventory, self.last_demand, self.lead_time, self.time_indicator], dtype=np.float32)
        
        done = self.current_step >= self.max_steps
        info = {"demand": demand, "order_qty": order_qty, "new_inventory": self.inventory, "cost": total_cost}
        return self.state, reward, done, info

    def render(self, mode="human"):
        print(f"Step: {self.current_step} | Inventory: {self.inventory:.2f} | Last Demand: {self.last_demand:.2f}")