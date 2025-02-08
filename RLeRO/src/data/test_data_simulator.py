import numpy as np
import pandas as pd

def generate_test_data(n_periods=1000, seed=42):
    np.random.seed(seed)
    data = []
    
    # Base parameters (same as training)
    base_demand = 50
    base_lead_time = 2
    
    # Increased volatility for testing:
    demand_volatility = 15  # larger than training volatility
    lead_time_volatility = 2  # more variable lead times
    
    # For testing, dynamic events are more frequent and severe:
    event_probability = 0.01  # higher probability than training
    for t in range(n_periods):
        # Simulate a dynamic event: 0 means normal, 1 means event (e.g., supplier disruption)
        event = np.random.choice([0, 1], p=[1 - event_probability, event_probability])
        # In test data, events cause larger increases in demand and lead time.
        event_demand_increase = 0.2 * base_demand if event else 0
        event_lead_time_increase = 2 if event else 0
        
        # Sample demand and lead time from a normal distribution
        demand = max(0, np.random.normal(base_demand + event_demand_increase, demand_volatility))
        lead_time = max(1, np.random.normal(base_lead_time + event_lead_time_increase, lead_time_volatility))
        
        # Randomly initialize inventory (could be taken from a realistic distribution)
        inventory = np.random.randint(20, 100)
        
        data.append({
            'time': t,
            'inventory': inventory,
            'demand': demand,
            'lead_time': lead_time,
            'event': event
        })
    
    return pd.DataFrame(data)

