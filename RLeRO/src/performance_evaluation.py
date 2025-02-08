import pandas as pd
import numpy as np
from src.config import Config

class PerformanceEvaluator:
    def __init__(self, test_data, env_class, robust_optimizer, rl_model):
        """
        test_data: pandas DataFrame generated from generate_test_data()
        env_class: Your custom Gym environment class (e.g., InventoryEnv)
        robust_optimizer: Function that returns (order_quantity, cost) using Gurobi
        rl_model: Trained PPO model from Stable Baselines 3
        entropy_threshold: Threshold above which RLeRO triggers robust optimization
        delta_demand, delta_lead_time: Uncertainty parameters for robust optimization in the test setting
        """
        config = Config()
        
        # load test data
        self.test_data = test_data
        
        # load environment
        self.env_class = env_class
        
        # load model
        self.robust_optimizer = robust_optimizer
        self.rl_model = rl_model
        
        # load hyper-parameters
        self.entropy_threshold = config.entropy_threshold
        self.delta_demand = config.delta_D
        self.delta_lead_time = config.delta_L

    def compute_policy_entropy(self, observation):
        """Compute the policy entropy for a given observation.
           Assumes that rl_model.policy.get_distribution() returns a distribution object with an entropy() method.
        """
        import torch
        obs_tensor = torch.as_tensor(observation).float().unsqueeze(0)
        dist = self.rl_model.policy.get_distribution(obs_tensor)
        entropy = dist.entropy().mean().item()
        return entropy

    def evaluate_episode(self, method):
        """
        Run a simulation episode over the test dataset using one of the methods:
        - 'vanilla_rl': always use the RL policy
        - 'pure_ro': always use robust optimization
        - 'RLeRO': use RL unless policy entropy is above the threshold (in which case use RO)
        Returns a dictionary of performance metrics.
        """
        # Create a new environment instance for the episode using test_data.
        env = self.env_class(self.test_data)
        obs = env.reset()
        done = False

        total_cost = 0.0
        cost_list = []
        stockout_count = 0
        inventory_levels = []
        demands = []
        robust_decisions_triggered = 0
        event_periods = 0
        event_cost_total = 0.0

        while not done:
            # State: [inventory, demand, lead_time, event]
            inventory, est_demand, est_lead_time, event = obs

            # Choose action based on method
            if method == 'vanilla_rl':
                action, _ = self.rl_model.predict(obs, deterministic=True)
            elif method == 'pure_ro':
                action, _ = self.robust_optimizer(
                    inventory, est_demand, self.delta_demand,
                    est_lead_time, self.delta_lead_time
                )
                action = int(action) if action is not None else 0
            elif method == 'RLeRO':
                # Compute the policy entropy for the current observation
                entropy = self.compute_policy_entropy(obs)
                if entropy > self.entropy_threshold:
                    # When uncertain, use robust optimization.
                    action, _ = self.robust_optimizer(
                        inventory, est_demand, self.delta_demand,
                        est_lead_time, self.delta_lead_time
                    )
                    action = int(action) if action is not None else 0
                    robust_decisions_triggered += 1
                else:
                    action, _ = self.rl_model.predict(obs, deterministic=True)
            else:
                raise ValueError("Unknown method: choose from 'vanilla_rl', 'pure_ro', 'RLeRO'")
            
            # Take a step in the environment
            next_obs, reward, done, _ = env.step(action)
            # Reward is negative cost, so convert:
            cost = -reward
            total_cost += cost
            cost_list.append(cost)
            
            # Count a stockout if the resulting inventory (in next_obs[0]) is negative.
            if next_obs[0] < 0:
                stockout_count += 1
            
            inventory_levels.append(next_obs[0])
            demands.append(obs[1])
            
            # If a dynamic event occurred in this period, record the cost.
            if obs[3] == 1:
                event_periods += 1
                event_cost_total += cost
            
            obs = next_obs

        # Compute aggregated metrics.
        avg_cost = total_cost / len(cost_list) if cost_list else np.nan
        avg_inventory = np.mean(inventory_levels) if inventory_levels else np.nan
        total_demand = np.sum(demands) if demands else np.nan
        inventory_turnover = total_demand / avg_inventory if avg_inventory > 0 else np.nan
        avg_event_cost = event_cost_total / event_periods if event_periods > 0 else np.nan

        metrics = {
            'Total Cost': total_cost,
            'Average Cost per Period': avg_cost,
            'Stockout Frequency': stockout_count,
            'Average Inventory Level': avg_inventory,
            'Inventory Turnover': inventory_turnover,
            'Average Cost during Event Periods': avg_event_cost,
            'Robust Decisions Triggered': robust_decisions_triggered  # only relevant for RLeRO
        }
        return metrics

    def evaluate_all_methods(self):
        """Run evaluation episodes for each method and return a table (DataFrame) of results."""
        methods = ['vanilla_rl', 'pure_ro', 'RLeRO']
        results = {}
        for method in methods:
            metrics = self.evaluate_episode(method)
            results[method] = metrics
        df = pd.DataFrame(results).T  # each row corresponds to a method
        return df
