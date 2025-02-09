from src.config import Config
import gurobipy as gp
from gurobipy import GRB

def robust_order_quantity(inventory, 
                          estimated_demand, delta_demand, 
                          estimated_yield_rate, delta_yield_rate,
                          next_day_orders=0, second_day_orders=0, later_orders=0):
    """
    Solve for the robust production quantity using Gurobi.
    This function computes the production quantity Q that minimizes the worst-case cost,
    taking into account both uncertainty in demand and production yield rate.
    
    Args:
        inventory: On-hand inventory (after processing arrivals)
        estimated_demand: Expected demand per period
        delta_demand: Uncertainty in demand
        estimated_yield_rate: Expected yield rate (between 0 and 1)
        delta_yield_rate: Uncertainty in yield rate
        next_day_orders: Orders scheduled to complete next period
        second_day_orders: Orders scheduled to complete in two periods
        later_orders: Orders scheduled to complete in three or more periods
    """
    config = Config()
    
    # Worst-case yield rate (minimum yield)
    worst_case_yield_rate = max(0.5, estimated_yield_rate - delta_yield_rate)
    
    # Worst-case demand
    worst_case_demand = estimated_demand + delta_demand

    # Create a new Gurobi model
    m = gp.Model("robust_inventory")
    m.Params.OutputFlag = 0  # Run in silent mode

    # Decision variable: production quantity Q (integer)
    Q = m.addVar(vtype=GRB.INTEGER, name="Q", lb=0, ub=config.max_order)

    # Compute effective inventory
    effective_inventory = inventory + next_day_orders + second_day_orders + later_orders

    # Expected production output after yield losses
    expected_production = Q * worst_case_yield_rate
    
    # Inventory level after production completion and demand
    new_inventory = effective_inventory + expected_production - worst_case_demand

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