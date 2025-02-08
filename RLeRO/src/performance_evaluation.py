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
        self.config = Config()
        
        # load test data
        self.test_data = test_data
        
        # load environment
        self.env_class = env_class
        
        # load model
        self.robust_optimizer = robust_optimizer
        self.rl_model = rl_model
        
        # load hyper-parameters
        self.entropy_threshold = self.config.entropy_threshold
        self.delta_demand = self.config.delta_D
        self.delta_lead_time = self.config.delta_L

    def compute_policy_entropy(self, observation):
        """Compute the normalized policy entropy for a given observation."""
        import torch
        import math

        obs_tensor = torch.as_tensor(observation).float().unsqueeze(0)
        dist = self.rl_model.policy.get_distribution(obs_tensor)
        entropy = dist.entropy().mean().item()  # Actual entropy

        # Get the number of actions
        action_dim = self.env_class(self.test_data).action_space.n  # Get action dimension from environment

        # Theoretical maximum entropy for normalization
        max_entropy = math.log(action_dim) if action_dim > 1 else 1  # Avoid division by zero for single-action spaces

        # Normalize entropy to [0, 1]
        normalized_entropy = entropy / max_entropy

        return normalized_entropy


    def evaluate_episode(self, method):
        """
        Run a simulation episode over the test dataset using one of the methods:
        - 'pure_rl': always use the RL policy
        - 'pure_ro': always use robust optimization
        - 'RLeRO': use RL unless policy entropy is above the threshold (in which case use RO)
        Returns a dictionary of performance metrics.
        """
        # Create a new environment instance for the episode using test_data.
        env = self.env_class(self.test_data)
        obs, _ = env.reset()  # Unpack the observation and info
        done = False

        total_cost = 0.0
        cost_list = []
        stockout_count = 0
        inventory_levels = []
        demands = []
        robust_decisions_triggered = 0

        while not done:
            # Get observation - obs is already the correct format array [inventory, demand, lead_time, event]
            inventory = obs[0]
            est_demand = obs[1]
            est_lead_time = obs[2]
            next_day_orders = obs[3]
            second_day_orders = obs[4]
            later_orders = obs[5]

            # Choose action based on method
            if method == 'pure_rl':
                action, _ = self.rl_model.predict(obs, deterministic=True)
            elif method == 'pure_ro':
                action, _ = self.robust_optimizer(
                    inventory, est_demand, self.delta_demand,
                    est_lead_time, self.delta_lead_time,
                    next_day_orders, second_day_orders, later_orders
                )
                action = int(action) if action is not None else 0
            elif method == 'RLeRO':
                # Compute the policy entropy for the current observation
                entropy = self.compute_policy_entropy(obs)
                if entropy > self.entropy_threshold:
                    # When uncertain, use robust optimization.
                    action, _ = self.robust_optimizer(
                        inventory, est_demand, self.delta_demand,
                        est_lead_time, self.delta_lead_time,
                        next_day_orders, second_day_orders, later_orders
                    )
                    action = int(action) if action is not None else 0
                    robust_decisions_triggered += 1
                else:
                    action, _ = self.rl_model.predict(obs, deterministic=True)
            else:
                raise ValueError("Unknown method: choose from 'pure_rl', 'pure_ro', 'RLeRO'")
            
            # Take a step in the environment
            next_obs, reward, done, _, _ = env.step(action)
            # Reward is negative cost, so convert:
            cost = -reward
            total_cost += cost
            cost_list.append(cost)
            
            # Count a stockout if the resulting inventory (in next_obs[0]) is negative.
            if next_obs[0] < 0:
                stockout_count += 1
            
            inventory_levels.append(next_obs[0])
            demands.append(est_demand)
            
            obs = next_obs

        # Compute aggregated metrics.
        avg_cost = total_cost / len(cost_list) if cost_list else np.nan
        avg_inventory = np.mean(inventory_levels) if inventory_levels else np.nan
        total_demand = np.sum(demands) if demands else np.nan
        inventory_turnover = total_demand / avg_inventory if avg_inventory > 0 else np.nan

        metrics = {
            'Total Cost': total_cost,
            'Average Cost per Period': avg_cost,
            'Stockout Frequency': stockout_count,
            'Average Inventory Level': avg_inventory,
            'Inventory Turnover': inventory_turnover,
            'Robust Decisions Triggered': robust_decisions_triggered  # only relevant for RLeRO
        }
        return metrics

    def evaluate_all_methods(self):
        """Run evaluation episodes for each method and return a table (DataFrame) of results."""
        methods = ['pure_rl', 'pure_ro', 'RLeRO']
        results = {}
        for method in methods:
            metrics = self.evaluate_episode(method)
            results[method] = metrics
        df = pd.DataFrame(results).T  # each row corresponds to a method
        return df
