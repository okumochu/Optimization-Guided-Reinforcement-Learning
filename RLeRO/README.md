## Reinforcement Learning embeded with Robust Optimization (RLeRO): Concept Overview

**Reinforcement Learning embeded with Robust Optimization** is a novel paradigm that synergizes the strengths of **Mathematical Optimization Models** and **Reinforcement Learning (RL)**. By embedding mathematical programming techniques into RL frameworks, RLeRO addresses key challenges in achieving efficient, high-quality decision-making in complex, dynamic environments.

**Challenges**:
   - The problem need to be able to be formulated as a mathematical programming problem.
   - Identifying when and how much optimization guidance is needed without stifling the agent’s exploration.
   - Balancing computational costs, as optimization solvers can be resource-intensive, especially in real-time applications.

## Implementation Details

### Environment Design
The inventory management environment is implemented as a custom Gym environment with:

- **State Space (6 dimensions)**:
  - Current inventory level
  - Forecasted demand
  - Expected lead time
  - Pending orders in three time windows:
    - Orders arriving next day
    - Orders arriving day after tomorrow
    - Orders arriving later

- **Action Space**:
  - Discrete order quantities (0 to max_order)
  - Actions represent the number of units to order

### Components

1. **Reinforcement Learning (RL)**:
   - Uses Proximal Policy Optimization (PPO) algorithm
   - MLP policy network for action selection
   - Trained on simulated historical data
   - Optimizes for minimizing total costs

2. **Robust Optimization (RO)**:
   - Mathematical programming using Gurobi solver
   - Considers worst-case scenarios for:
     - Demand uncertainty (±δD)
     - Lead time uncertainty (±δL)
   - Accounts for pending orders in different time windows
   - Minimizes worst-case total cost

3. **RLeRO Integration**:
   - Hybrid decision-making system
   - Uses policy entropy as uncertainty measure
   - Switches to RO when RL policy uncertainty exceeds threshold
   - Maintains benefits of both approaches:
     - RL's adaptability in normal conditions
     - RO's robustness in uncertain situations

### Performance Metrics
System evaluation based on:
- Total operational cost
- Average cost per period
- Stockout frequency
- Average inventory level
- Inventory turnover
- Number of robust decisions triggered

### Data Generation
- Simulated data with configurable parameters:
  - Base demand and lead time
  - Volatility levels
  - Different modes for training and testing
- Supports both training and evaluation scenarios

This implementation provides a flexible framework for studying the integration of RL and RO in inventory management, with particular focus on handling uncertainty through hybrid decision-making.