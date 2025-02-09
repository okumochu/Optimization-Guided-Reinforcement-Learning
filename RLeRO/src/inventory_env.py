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
        
        # Updated Observation:
        # Now only [inventory, forecast_demand, forecast_yield_rate]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: Production quantity (integer) between 0 and max_order.
        self.action_space = spaces.Discrete(config.max_order + 1)
        
        # Cost parameters.
        self.holding_cost = config.C_h
        self.shortage_cost = config.C_s
        self.production_cost = config.C_o
        
        # Pending productions will be stored as a list of tuples: (planned_quantity, yield_rate)
        self.pending_productions = []
    
    def process_pending_orders(self):
        """
        Process pending productions at the beginning of the period.
        Apply yield rate to determine actual quantity received.
        """
        new_pending_productions = []
        for planned_qty, yield_rate in self.pending_productions:
            actual_qty = int(planned_qty * yield_rate)  # Apply yield rate
            self.inventory += actual_qty  # Production arrives now
        self.pending_productions = new_pending_productions
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        state = self.data.iloc[self.current_step]
        self.inventory = state['inventory']  # Initialize on-hand inventory from data.
        observation = np.array([
            self.inventory,
            state['demand'],
            state['yield_rate']
        ], dtype=np.float32)
        return observation, {}

    def step(self, action):
        # 1. Process pending productions that are completed
        self.process_pending_orders()
        
        # 2. Get the current period's state from the offline data
        state = self.data.iloc[self.current_step]
        demand = state['demand']
        yield_rate = state['yield_rate']        
        
        # 3. The agent's action is the production quantity
        prod_qty = action
        
        # 4. Simulate demand consumption for the current period
        actual_demand = demand  
        self.inventory -= actual_demand
        
        # 5. Calculate costs based on the post-demand inventory
        holding_cost = self.holding_cost * max(self.inventory, 0)
        shortage_cost = self.shortage_cost * max(-self.inventory, 0)
        production_cost = self.production_cost * prod_qty
        total_cost = holding_cost + shortage_cost + production_cost
        reward = -total_cost
        
        # 6. Start new production. It will be completed with the given yield rate
        if prod_qty > 0:
            self.pending_productions.append((prod_qty, yield_rate))
        
        # 7. Advance the simulation.
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        # 8. Construct the next observation.
        if not done:
            next_state = self.data.iloc[self.current_step]
            next_observation = np.array([
                self.inventory,
                next_state['demand'],
                next_state['yield_rate']
            ], dtype=np.float32)
        else:
            next_observation = np.array([
                self.inventory,
                0,  # No further demand.
                0   # No further yield rate.
            ], dtype=np.float32)
            
        return next_observation, reward, done, False, {}
