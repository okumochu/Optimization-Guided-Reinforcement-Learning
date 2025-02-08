from src.config import Config
import gurobipy as gp
from gurobipy import GRB

def robust_order_quantity(inventory, 
                          estimated_demand, delta_demand, 
                          estimated_lead_time, delta_lead_time,
                          next_day_orders=0, second_day_orders=0, later_orders=0):
    """
    Solve for the robust order quantity using Gurobi.
    This function computes the order quantity Q that minimizes the worst-case cost
    over the lead time period, taking into account both uncertainty in demand and lead time.
    
    Args:
        inventory: On-hand inventory (after processing arrivals).
        estimated_demand: Expected demand per period.
        delta_demand: Uncertainty in demand.
        estimated_lead_time: Expected lead time (in periods).
        delta_lead_time: Uncertainty in lead time.
        next_day_orders: Orders scheduled to arrive next period.
        second_day_orders: Orders scheduled to arrive in two periods.
        later_orders: Orders scheduled to arrive in three or more periods.
        
    Returns:
        robust_Q: The optimal order quantity under the worst-case scenario.
        robust_cost: The corresponding worst-case cost.
    """
    config = Config()
    
    # Worst-case lead time.
    worst_case_lead_time = estimated_lead_time + delta_lead_time
    
    # Worst-case demand per period.
    worst_case_demand_rate = estimated_demand + delta_demand
    worst_case_demand_during_leadtime = worst_case_demand_rate * worst_case_lead_time

    # Create a new Gurobi model.
    m = gp.Model("robust_inventory")
    m.Params.OutputFlag = 0  # Run in silent mode.

    # Decision variable: order quantity Q (integer).
    Q = m.addVar(vtype=GRB.INTEGER, name="Q", lb=0, ub=config.max_order)

    # Compute effective inventory.
    # (Since arrivals have already been processed, effective inventory = current inventory + pending orders.)
    effective_inventory = inventory + next_day_orders + second_day_orders + later_orders

    # Inventory level after receiving the order Q and after worst-case demand during the lead time.
    new_inventory = effective_inventory + Q - worst_case_demand_during_leadtime

    # Introduce auxiliary variables for holding (pos) and shortage (neg) components.
    pos = m.addVar(vtype=GRB.CONTINUOUS, name="pos", lb=0)
    neg = m.addVar(vtype=GRB.CONTINUOUS, name="neg", lb=0)
    
    # Constraints to linearize the piecewise linear cost:
    # pos = max(new_inventory, 0) and neg = max(-new_inventory, 0)
    m.addConstr(pos >= new_inventory)
    m.addConstr(pos >= 0)
    m.addConstr(neg >= -new_inventory)
    m.addConstr(neg >= 0)
    
    # Total cost: holding cost + shortage cost + ordering cost.
    total_cost = config.C_h * pos + config.C_s * neg + config.C_o * Q
    m.setObjective(total_cost, GRB.MINIMIZE)
    
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        robust_Q = int(Q.X)
        robust_cost = m.objVal
        return robust_Q, robust_cost
    else:
        return None, None