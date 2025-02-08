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
        
        # Observation: [inventory, forecast_demand, forecast_lead_time, next_day_orders, second_day_orders, later_orders]
        # Note: Inventory can go negative (representing shortages), so we allow negative values.
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: Order quantity (integer) between 0 and max_order.
        self.action_space = spaces.Discrete(config.max_order + 1)
        
        # Cost parameters.
        self.holding_cost = config.C_h
        self.shortage_cost = config.C_s
        self.ordering_cost = config.C_o
        
        # Pending orders will be stored as a list of tuples: (quantity, remaining_lead_time)
        self.pending_orders = []
    
    def process_pending_orders(self):
        """
        Process pending orders at the beginning of the period.
        Orders with remaining_time <= 1 are delivered now (and added to the on-hand inventory);
        others have their remaining time decremented by 1.
        """
        new_pending_orders = []
        for qty, remaining_time in self.pending_orders:
            if remaining_time <= 1:
                self.inventory += qty  # Order arrives now.
            else:
                new_pending_orders.append((qty, remaining_time - 1))
        self.pending_orders = new_pending_orders
    
    def get_pending_orders_by_time(self):
        """
        Break down pending orders by their remaining lead time.
        After processing arrivals, pending orders are those that are still not arrived.
        They are classified as:
          - next_day_orders: orders with remaining_time == 1
          - second_day_orders: orders with remaining_time == 2
          - later_orders: orders with remaining_time >= 3
        """
        next_day_orders = 0
        second_day_orders = 0
        later_orders = 0
        for qty, remaining_time in self.pending_orders:
            if remaining_time == 1:
                next_day_orders += qty
            elif remaining_time == 2:
                second_day_orders += qty
            else:
                later_orders += qty
        return next_day_orders, second_day_orders, later_orders

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.pending_orders = []  # Start with no pending orders.
        state = self.data.iloc[self.current_step]
        self.inventory = state['inventory']  # Initialize on-hand inventory from data.
        next_day_orders, second_day_orders, later_orders = self.get_pending_orders_by_time()
        observation = np.array([
            self.inventory,
            state['demand'],
            state['lead_time'],
            next_day_orders,
            second_day_orders,
            later_orders
        ], dtype=np.float32)
        return observation, {}

    def step(self, action):
        # 1. Process pending orders that are due to arrive at the beginning of the period.
        self.process_pending_orders()
        
        # 2. Get the current period's state from the offline data.
        state = self.data.iloc[self.current_step]
        demand = state['demand']
        lead_time = state['lead_time']        
        
        # 3. The agent's action is the order quantity.
        order_qty = action
        
        # 4. Simulate demand consumption for the current period.
        actual_demand = demand  
        self.inventory -= actual_demand
        
        # 5. Calculate costs based on the post-demand inventory.
        holding_cost = self.holding_cost * max(self.inventory, 0)
        shortage_cost = self.shortage_cost * max(-self.inventory, 0)
        ordering_cost = self.ordering_cost * order_qty
        total_cost = holding_cost + shortage_cost + ordering_cost
        reward = -total_cost
        
        # 6. Place the new order. It will be delivered after the lead time.
        if order_qty > 0:
            self.pending_orders.append((order_qty, lead_time))
        
        # 7. Advance the simulation.
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        # 8. Construct the next observation.
        if not done:
            next_state = self.data.iloc[self.current_step]
            next_day_orders, second_day_orders, later_orders = self.get_pending_orders_by_time()
            next_observation = np.array([
                self.inventory,
                next_state['demand'],
                next_state['lead_time'],
                next_day_orders,
                second_day_orders,
                later_orders
            ], dtype=np.float32)
        else:
            next_observation = np.array([
                self.inventory,
                0,  # No further demand.
                0,  # No further lead time.
                0,
                0,
                0
            ], dtype=np.float32)
            
        return next_observation, reward, done, False, {}
