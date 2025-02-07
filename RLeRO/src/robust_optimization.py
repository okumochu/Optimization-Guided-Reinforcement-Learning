import gurobipy as gp
from gurobipy import GRB

def robust_optimization_gurobi(I_t, D_hat, delta, C_h, C_s, C_o):
    """
    Solves the robust optimization problem using Gurobi.
    
    The problem is defined as:
        min_{Q >= 0} max_{D in U} [ C_h * pos(I_t + Q - D) + C_s * pos(D - (I_t + Q)) + C_o * Q ]
    with the uncertainty set U = [D_hat - delta, D_hat + delta].
    
    This formulation is linearized by introducing auxiliary variables:
      - y1, y2 for the case D = D_hat - delta
      - z1, z2 for the case D = D_hat + delta
    such that:
        cost1 = C_h * y1 + C_s * y2 + C_o * Q  (for D = D_hat - delta)
        cost2 = C_h * z1 + C_s * z2 + C_o * Q  (for D = D_hat + delta)
    and an auxiliary variable t is forced to be at least as large as both cost1 and cost2.
    The model minimizes t.
    """
    model = gp.Model("robust_inventory")
    model.setParam('OutputFlag', 0)  # suppress output

    # Decision variable for order quantity (Q) and auxiliary variable (t)
    Q = model.addVar(lb=0, name="Q")
    t = model.addVar(lb=-GRB.INFINITY, name="t")
    
    # Auxiliary variables for lower extreme (D_hat - delta)
    y1 = model.addVar(lb=0, name="y1")
    y2 = model.addVar(lb=0, name="y2")
    
    # Auxiliary variables for upper extreme (D_hat + delta)
    z1 = model.addVar(lb=0, name="z1")
    z2 = model.addVar(lb=0, name="z2")
    
    model.update()
    
    # For D = D_hat - delta:
    # y1 >= I_t + Q - (D_hat - delta)
    model.addConstr(y1 >= I_t + Q - (D_hat - delta), "y1_constraint")
    # y2 >= (D_hat - delta) - (I_t + Q)
    model.addConstr(y2 >= (D_hat - delta) - (I_t + Q), "y2_constraint")
    
    # For D = D_hat + delta:
    # z1 >= I_t + Q - (D_hat + delta)
    model.addConstr(z1 >= I_t + Q - (D_hat + delta), "z1_constraint")
    # z2 >= (D_hat + delta) - (I_t + Q)
    model.addConstr(z2 >= (D_hat + delta) - (I_t + Q), "z2_constraint")
    
    # Define cost expressions at both extremes:
    cost1 = C_h * y1 + C_s * y2 + C_o * Q
    cost2 = C_h * z1 + C_s * z2 + C_o * Q
    
    # Ensure t is at least as large as both costs.
    model.addConstr(t >= cost1, "t_ge_cost1")
    model.addConstr(t >= cost2, "t_ge_cost2")
    
    # Set the objective to minimize t.
    model.setObjective(t, GRB.MINIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        Q_val = Q.X
    else:
        print("Gurobi did not find an optimal solution; defaulting to RL action.")
        Q_val = 0.0
    return Q_val