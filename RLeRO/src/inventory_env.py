import gym
from gym import spaces
import numpy as np
from src.config import Config

class InventoryEnv(gym.Env):
    def __init__(self, offline_data):
        self.data = offline_data.reset_index(drop=True)
        self.current_step = 0
        config = Config()
        # Define action and observation space
        # Action: integer order quantity between 0 and max_order
        self.action_space = spaces.Discrete(config.max_order + 1)
        # Observation: [inventory, demand, lead_time, event]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Cost parameters
        self.holding_cost = config.C_h
        self.shortage_cost = config.C_s
        self.ordering_cost = config.C_o

    def reset(self):
        self.current_step = 0
        # Reset inventory to a value from the offline data
        state = self.data.iloc[self.current_step]
        return np.array([state['inventory'], state['demand'], state['lead_time'], state['event']], dtype=np.float32)
    
    def step(self, action):
        # Current state from offline data
        state = self.data.iloc[self.current_step]
        inventory = state['inventory']
        demand = state['demand']
        lead_time = state['lead_time']
        event = state['event']
        
        # Action is order quantity Q
        order_qty = action
        
        # Calculate next inventory level (ignoring lead time for simplicity in state update)
        new_inventory = inventory + order_qty - demand
        
        # Cost calculation: holding, shortage, and ordering cost
        holding_cost = self.holding_cost * max(new_inventory, 0)
        shortage_cost = self.shortage_cost * max(-new_inventory, 0)
        order_cost = self.ordering_cost * order_qty
        total_cost = holding_cost + shortage_cost + order_cost
        reward = -total_cost
        
        # For simplicity, update the state with new inventory and sample next period from offline data
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        if not done:
            next_state_raw = self.data.iloc[self.current_step]
            next_state = np.array([new_inventory, next_state_raw['demand'], next_state_raw['lead_time'], next_state_raw['event']], dtype=np.float32)
        else:
            next_state = np.array([new_inventory, 0, 0, 0], dtype=np.float32)
            
        return next_state, reward, done, {}
