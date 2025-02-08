import numpy as np
import pandas as pd

def generate_offline_data(n_periods=1000, seed=42):
    np.random.seed(seed)
    data = []
    
    # Base parameters for demand and lead time
    base_demand = 50
    demand_volatility = 10  # standard deviation
    base_lead_time = 2      # days
    lead_time_volatility = 1

    # Dynamic event indicator: 0 for normal, 1 for event (e.g., supplier disruption)
    # In training, events are mild: they slightly increase demand and lead time.
    for t in range(n_periods):
        # simulate a dynamic event (rare but consistent in training)
        event = np.random.choice([0, 1], p=[0.9, 0.1])
        # if an event happens, add a fixed increase during training (e.g., +10% demand, +1 day lead time)
        event_demand_increase = 0.1 * base_demand if event else 0
        event_lead_time_increase = 1 if event else 0
        
        demand = max(0, np.random.normal(base_demand + event_demand_increase, demand_volatility))
        lead_time = max(1, np.random.normal(base_lead_time + event_lead_time_increase, lead_time_volatility))
        inventory = np.random.randint(20, 100)  # randomly initialize inventory
        
        # Record each time period's data
        data.append({
            'time': t,
            'inventory': inventory,
            'demand': demand,
            'lead_time': lead_time,
            'event': event
        })
    
    df = pd.DataFrame(data)
    return df
