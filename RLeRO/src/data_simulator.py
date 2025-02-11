import numpy as np
import pandas as pd
from src.config import Config

def generate_data(n_periods, mode, seed=42):
    config = Config()
    np.random.seed(seed)
    data = []
    
    if mode == "train":
        demand_volatility = config.train_demand_volatility
        yield_rate_volatility = config.train_yield_rate_volatility # 2.5% variation in yield rate
    elif mode == "test":
        demand_volatility = config.test_demand_volatility
        yield_rate_volatility = config.test_yield_rate_volatility   # 10% variation in yield rate
    
    demand_perturbation = config.base_demand
    yield_rate_perturbation = config.base_yield_rate
    
    for t in range(n_periods):
        # Ensure demand stays between 0 and max_demand
        demand = np.random.normal(demand_perturbation, demand_volatility)
        demand = np.clip(demand, 0, config.max_demand)
        demand_perturbation = demand
        
        # Ensure yield rate stays between 0 and 1
        yield_rate = np.random.normal(yield_rate_perturbation, yield_rate_volatility)
        yield_rate = np.clip(yield_rate, config.min_yield_rate, config.max_yield_rate)
        yield_rate_perturbation = yield_rate

        
        inventory = np.random.randint(0, config.max_inventory)
        
        data.append({
            'time': t,
            'inventory': inventory,
            'demand': demand,
            'yield_rate': yield_rate,
        })
    
    return pd.DataFrame(data)

