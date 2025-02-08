## Reinforcement Learning embeded with Robust Optimization (RLeRO): Concept Overview

**Reinforcement Learning embeded with Robust Optimization** is a novel paradigm that synergizes the strengths of **Mathematical Optimization Models** and **Reinforcement Learning (RL)**. By embedding mathematical programming techniques into RL frameworks, RLeRO addresses key challenges in achieving efficient, high-quality decision-making in complex, dynamic environments.

**Challenges**:
   - The problem need to be able to be formulated as a mathematical programming problem.
   - Identifying when and how much optimization guidance is needed without stifling the agent’s exploration.
   - Balancing computational costs, as optimization solvers can be resource-intensive, especially in real-time applications.


## Implementation Details

### **RLeRO for Inventory Management with Dynamic Events, Lead Time & Demand Uncertainty**

#### **1. Problem Overview:**

The goal is to manage inventory under uncertain demand and lead times. Reinforcement Learning (RL) using Proximal Policy Optimization (PPO) from Stable Baselines 3 will handle regular decision-making, while Robust Optimization (RO), solved via Gurobi, will intervene when RL encounters unfamiliar scenarios.

---

#### **2. Key Enhancements:**

1. **Dynamic Events:**  
   - RL can observe dynamic external events (e.g., supplier disruptions, sudden policy changes, or unexpected market shifts).  
   - During training, these events typically maintain consistent patterns. However, in deployment, they vary, causing RL uncertainty. This triggers RO to make robust decisions.

2. **Two Sources of Uncertainty:**  
   - **Demand Uncertainty:** Fluctuations in customer demand.  
   - **Lead Time Uncertainty:** Variability in supplier delivery times.

3. **Offline Data:**  
   - RL will train and evaluate on pre-generated offline datasets, reflecting realistic inventory, demand, lead time, and event patterns.

---

#### **3. Problem Setup:**

- **State $s$:**  
  Includes current inventory levels, recent demand history, lead time estimates, and dynamic event indicators.

- **Action $a$:**  
  Order quantity for the next period.

- **Reward $r$:**  
  Negative of total cost, combining:
  - Holding costs $C_h$
  - Stockout penalties $C_s$
  - Ordering costs $C_o$
  - Event-related costs (optional, depending on the dynamic event impact)

- **Transition Dynamics:**  
  $I_{t+1} = I_t + Q_t - D_t$  
  $L_{t+1} = f(L_t, \text{event}_t)$ (lead time influenced by dynamic events)

---

#### **4. Dynamic Events:**

- **Examples:** Supplier strikes, transport delays, sudden demand surges.
- **Training Behavior:** Dynamic events are relatively consistent (e.g., always mild delays).
- **Deployment Behavior:** Events deviate from training norms, causing RL uncertainty, triggering RO.

---

#### **5. Reinforcement Learning Component (PPO):**

- **Algorithm:** PPO from Stable Baselines 3.  
- **Policy Entropy Measurement:**  
  $H(\pi(s, a)) = -\sum_a \pi(a|s) \log \pi(a|s) $  
- **Trigger Condition:**  
  If $H(\pi(s, a)) > \theta$, RO is activated.

---

#### **6. Robust Optimization Component (Gurobi):**

- **Objective:**  
  $ \min_{Q} \max_{D \in \mathcal{U}_D, L \in \mathcal{U}_L} \left( C_h \cdot \max(0, I_t + Q - D) + C_s \cdot \max(0, D - (I_t + Q)) + C_o \cdot Q \right) $

- **Uncertainty Sets:**  
  - $ \mathcal{U}_D = [\hat{D} - \delta_D, \hat{D} + \delta_D] $  
  - $ \mathcal{U}_L = [\hat{L} - \delta_L, \hat{L} + \delta_L] $  
  These sets are derived from offline data variability.

---

#### **7. Offline Data Design:**

- **Structure:**  
  - Time-stamped records of demand, lead time, inventory levels, dynamic events, and order decisions.
- **Generation:**  
  - Simulate various demand patterns (seasonal, random spikes).  
  - Lead times influenced by dynamic event scenarios.  
  - Ensure consistency in event behavior for training, with variability for testing.

---

#### **8. Workflow of RLeRO:**

1. **Observation:** RL observes current state $s_t$ (inventory, demand history, events).  
2. **Decision:** RL proposes action $a_t$ with entropy $H(\pi(s_t, a_t))$.  
3. **Trigger Check:**  
   - If $H(\pi(s_t, a_t)) \leq \theta$, execute RL’s decision.  
   - If $H(\pi(s_t, a_t)) > \theta$, trigger RO for robust decision-making.  
4. **Execution:** Apply the chosen action, update the system state based on offline data.  
5. **Learning:** RL updates its policy based on the observed rewards.

---

#### **9. Evaluation Metrics:**

- Total Cost (across demand, lead time, and event variations)
- Stockout Frequency and Severity
- Robustness of Decisions under Unfamiliar Events
- RL’s Adaptability to Dynamic Events
- Comparison with pure RL and pure RO baselines

