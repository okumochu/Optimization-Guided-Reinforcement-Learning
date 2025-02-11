import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.config import Config

class InventoryEnv(gym.Env):
    def __init__(self, offline_data):
        self.data = offline_data.reset_index(drop=True)
        self.current_step = 0
        config = Config()
        self.config = config
        
        # Initially, the inventory will be set from the data at reset.
        self.inventory = None  
        
        # History length for demand and yield rate
        self.history_length = 7
        
        # Initialize history buffers with zeros
        self.demand_history = np.zeros(self.history_length)
        self.yield_rate_history = np.zeros(self.history_length)
        
        # Updated Observation Space:
        # [inventory, 7 demands, 7 yield rates]
        obs_dim = 1 + 2 * self.history_length  # 1 for inventory + 7 demands + 7 yield rates
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * obs_dim, dtype=np.float32),
            high=np.array([np.inf] * obs_dim, dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: Production quantity (integer) between 0 and max_order.
        # Ensure action space matches the length of production_level_option
        self.action_space = spaces.Discrete(len(config.production_level_option))
        
        # Cost parameters.
        self.holding_cost = config.C_h
        self.shortage_cost = config.C_s
        self.production_cost = config.C_o
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        state = self.data.iloc[self.current_step]
        self.inventory = state['inventory']  # Initialize on-hand inventory from data.
        
        # Reset history buffers
        self.demand_history = np.zeros(self.history_length)
        self.yield_rate_history = np.zeros(self.history_length)
        
        # Initialize first observation
        observation = np.concatenate([
            [self.inventory],
            self.demand_history,
            self.yield_rate_history
        ]).astype(np.float32)
        
        return observation, {}

    def step(self, action):
        # Get the current period's state from the offline data
        state = self.data.iloc[self.current_step]
        current_demand = state['demand']
        current_yield_rate = state['yield_rate']        
        
        # The agent's action is the production quantity
        prod_qty = self.config.production_level_option[action]
        
        # Apply yield rate immediately to production and add to inventory
        actual_production = int(prod_qty * current_yield_rate)
        self.inventory += actual_production
        
        # Simulate demand consumption for the current period
        self.inventory -= current_demand
        
        # Calculate costs based on the post-demand inventory
        holding_cost = self.holding_cost * max(self.inventory, 0)
        shortage_cost = self.shortage_cost * max(-self.inventory, 0)
        production_cost = self.production_cost * prod_qty
        total_cost = holding_cost + shortage_cost + production_cost
        reward = -total_cost
        
        # Update history buffers
        self.demand_history = np.roll(self.demand_history, -1)
        self.demand_history[-1] = current_demand
        
        self.yield_rate_history = np.roll(self.yield_rate_history, -1)
        self.yield_rate_history[-1] = current_yield_rate
        
        # Advance the simulation.
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        # Construct the next observation.
        if not done:
            next_state = self.data.iloc[self.current_step]
            next_observation = np.concatenate([
                [self.inventory],
                self.demand_history,
                self.yield_rate_history
            ]).astype(np.float32)
        else:
            next_observation = np.concatenate([
                [self.inventory],
                np.zeros(self.history_length),  # No further demands
                np.zeros(self.history_length)   # No further yield rates
            ]).astype(np.float32)
            
        return next_observation, reward, done, False, {}
