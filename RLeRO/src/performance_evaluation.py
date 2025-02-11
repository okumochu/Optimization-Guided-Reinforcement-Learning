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
        
        Hyper-parameters:
         - entropy_threshold: Threshold above which RLeRO triggers robust optimization
         - delta_demand, delta_lead_time: Uncertainty parameters for robust optimization in the test setting
           (Note: the `delta_lead_time` parameter is used for the yield_rate uncertainty).
        """
        self.config = Config()
        
        # load test data
        self.test_data = test_data
        
        # load environment
        self.env_class = env_class
        
        # load optimizers and models
        self.robust_optimizer = robust_optimizer
        self.rl_model = rl_model
        
        # load hyper-parameters
        self.entropy_threshold = self.config.entropy_threshold
        self.delta_demand = self.config.delta_D
        self.delta_yield_rate = self.config.delta_Y

    def compute_policy_entropy(self, observation):
        """Compute the normalized policy entropy for a given observation."""
        import torch
        import math

        obs_tensor = torch.as_tensor(observation).float().unsqueeze(0)
        dist = self.rl_model.policy.get_distribution(obs_tensor)
        entropy = dist.entropy().mean().item()  # Actual entropy

        # Get the number of actions from the environment.
        action_dim = self.env_class(self.test_data).action_space.n

        # Theoretical maximum entropy for normalization
        max_entropy = math.log(action_dim) if action_dim > 1 else 1

        # Normalize entropy to [0, 1]
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def evaluate_episode(self, method):
        """
        Run a simulation episode over the test dataset using one of the methods:
        - 'pure_rl': Always use the RL policy.
        - 'pure_ro': Always use robust optimization.
        - 'RLeRO': Use RL unless the policy entropy is above the threshold (in which case use RO).
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
            # Observation now includes history: [inventory, demand_history(7), yield_rate_history(7)]
            inventory = obs[0]
            est_demand = obs[1]  # Using the most recent demand from history
            est_yield_rate = obs[8]  # Using the most recent yield rate from history

            # Choose action based on method
            if method == 'pure_rl':
                action, _ = self.rl_model.predict(obs, deterministic=True)
            elif method == 'pure_ro':
                ro_action, _ = self.robust_optimizer(
                    inventory, est_demand, self.delta_demand,
                    est_yield_rate, self.delta_yield_rate
                )
                # Convert RO quantity to discrete action index
                action = min(range(len(self.config.production_level_option)), 
                           key=lambda i: abs(self.config.production_level_option[i] - ro_action))
            elif method == 'RLeRO':
                entropy = self.compute_policy_entropy(obs)
                if entropy > self.entropy_threshold:
                    ro_action, _ = self.robust_optimizer(
                        inventory, est_demand, self.delta_demand,
                        est_yield_rate, self.delta_yield_rate
                    )
                    # Convert RO quantity to discrete action index
                    action = min(range(len(self.config.production_level_option)), 
                               key=lambda i: abs(self.config.production_level_option[i] - ro_action))
                    robust_decisions_triggered += 1
                else:
                    action, _ = self.rl_model.predict(obs, deterministic=True)
            else:
                raise ValueError("Unknown method: choose from 'pure_rl', 'pure_ro', 'RLeRO'")
            
            # Take a step in the environment.
            next_obs, reward, done, _, _ = env.step(action)
            
            # Reward is negative cost; convert to cost.
            cost = -reward
            total_cost += cost
            cost_list.append(cost)
            
            # Count a stockout if the resulting inventory is negative.
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
            'Robust Decisions Triggered': robust_decisions_triggered  # Only relevant for RLeRO.
        }
        return metrics

    def evaluate_all_methods(self):
        """Run evaluation episodes for each method and return a table (DataFrame) of results."""
        methods = ['pure_rl', 'pure_ro', 'RLeRO']
        results = {}
        for method in methods:
            metrics = self.evaluate_episode(method)
            results[method] = metrics
        df = pd.DataFrame(results).T  # Each row corresponds to a method.
        return df

    def plot_yield_demand_ro_triggers(self):
        """
        Plot yield rate and demand over time, marking points where RO was triggered.
        Returns the figure for optional further customization.
        """
        import matplotlib.pyplot as plt

        # Create a new environment instance and run an episode with RLeRO
        env = self.env_class(self.test_data)
        obs, _ = env.reset()
        done = False
        
        demands = []
        yield_rates = []
        ro_triggered_steps = []
        step = 0
        
        while not done:
            # Get current demand and yield rate from observation
            demands.append(obs[1])  # Most recent demand from history
            yield_rates.append(obs[8])  # Most recent yield rate from history
            
            # Check if RO should be triggered
            entropy = self.compute_policy_entropy(obs)
            if entropy > self.entropy_threshold:
                ro_triggered_steps.append(step)
            
            # Take step with RLeRO policy
            action, _ = self.rl_model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            step += 1
        
        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot demand on primary y-axis
        ax1.plot(demands, color='blue', label='Demand')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Demand', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create secondary y-axis for yield rate
        ax2 = ax1.twinx()
        ax2.plot(yield_rates, color='red', label='Yield Rate')
        ax2.set_ylabel('Yield Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add vertical lines for RO triggers
        for ro_step in ro_triggered_steps:
            plt.axvline(x=ro_step, color='green', alpha=0.3, linestyle='--')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Add text for RO triggers
        plt.text(0.02, 0.98, f'RO Triggered: {len(ro_triggered_steps)} times', 
                transform=ax1.transAxes, verticalalignment='top')
        
        plt.title('Demand and Yield Rate Over Time\nwith RO Trigger Points')
        plt.tight_layout()
        
        return fig
