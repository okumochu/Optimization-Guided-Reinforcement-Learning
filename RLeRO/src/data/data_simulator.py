import numpy as np
import pandas as pd

def generate_data(n_periods, mode, seed=42):
    np.random.seed(seed)
    data = []
    
    # Base parameters
    base_demand = 50
    base_yield_rate = 0.9  # 90% yield rate on average
    
    if mode == "train":
        demand_volatility = 10
        yield_rate_volatility = 0.05  # 5% variation in yield rate
    elif mode == "test":
        demand_volatility = 20
        yield_rate_volatility = 0.1   # 10% variation in yield rate
    
    for t in range(n_periods):
        demand = max(0, np.random.normal(base_demand, demand_volatility))
        # Ensure yield rate stays between 0 and 1
        yield_rate = np.clip(
            np.random.normal(base_yield_rate, yield_rate_volatility),
            0.5,  # minimum 50% yield
            1.0   # maximum 100% yield
        )
        
        inventory = np.random.randint(20, 100)
        
        data.append({
            'time': t,
            'inventory': inventory,
            'demand': demand,
            'yield_rate': yield_rate,
        })
    
    return pd.DataFrame(data)

