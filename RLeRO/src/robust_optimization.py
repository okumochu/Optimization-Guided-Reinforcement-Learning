import gurobipy as gp
from gurobipy import GRB

def robust_order_quantity(inventory, 
                          estimated_demand, delta_demand, 
                          estimated_lead_time, delta_lead_time,
                          holding_cost=1.0, shortage_cost=5.0, ordering_cost=2.0,
                          max_order=100):
    """
    Solve for the robust order quantity using Gurobi.
    Here, we use worst-case scenarios from the uncertainty sets:
      - Worst-case demand: highest demand (i.e., estimated_demand + delta_demand)
      - Worst-case lead time: assume it impacts cost indirectly (if incorporated).
    
    For simplicity, we consider only worst-case demand in the cost function.
    """
    worst_case_demand = estimated_demand + delta_demand

    # Create a new model
    m = gp.Model("robust_inventory")
    m.Params.OutputFlag = 0  # silent mode

    # Decision variable: order quantity Q (integer decision)
    Q = m.addVar(vtype=GRB.INTEGER, name="Q", lb=0, ub=max_order)

    # Calculate inventory after ordering and worst-case demand consumption
    new_inventory = inventory + Q - worst_case_demand

    # Modeling the cost components:
    # Use linear approximations: holding cost when new_inventory > 0 and shortage cost when new_inventory < 0
    # We introduce auxiliary variables to represent these costs.
    pos = m.addVar(vtype=GRB.CONTINUOUS, name="pos", lb=0)
    neg = m.addVar(vtype=GRB.CONTINUOUS, name="neg", lb=0)
    
    # Constraints to linearize max(0, new_inventory) and max(0, -new_inventory)
    m.addConstr(pos >= new_inventory)
    m.addConstr(pos >= 0)
    m.addConstr(neg >= -new_inventory)
    m.addConstr(neg >= 0)
    
    # Total cost: holding cost + shortage cost + ordering cost
    total_cost = holding_cost * pos + shortage_cost * neg + ordering_cost * Q
    m.setObjective(total_cost, GRB.MINIMIZE)
    
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        robust_Q = Q.X
        robust_cost = m.objVal
        return robust_Q, robust_cost
    else:
        return None, None
