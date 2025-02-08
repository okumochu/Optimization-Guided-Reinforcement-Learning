import numpy as np
import pandas as pd

def generate_data(n_periods, mode, seed=42):
    np.random.seed(seed)
    data = []
    
    # Base parameters (same as training)
    base_demand = 50
    base_lead_time = 2
    
    if mode == "train":
        # Increased volatility for testing:
        demand_volatility = 10  # larger than training volatility
        lead_time_volatility = 1  # more variable lead times
    elif mode == "test":
        demand_volatility = 20
        lead_time_volatility = 2
    
    for t in range(n_periods):
        demand = max(0, np.random.normal(base_demand, demand_volatility))
        lead_time = max(1, np.random.normal(base_lead_time, lead_time_volatility))
        
        # Randomly initialize inventory (could be taken from a realistic distribution)
        inventory = np.random.randint(20, 100)
        
        data.append({
            'time': t,
            'inventory': inventory,
            'demand': demand,
            'lead_time': lead_time,
        })
    
    return pd.DataFrame(data)

