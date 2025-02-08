from src.config import Config
from src.inventory_env import InventoryEnv
from stable_baselines3 import PPO

def train_ppo_model(train_data):
    config = Config()
    env_train = InventoryEnv(offline_data=train_data)
    rl_model = PPO('MlpPolicy', env_train, verbose=0, tensorboard_log=config.training_log_dir)
    rl_model.learn(total_timesteps=config.n_steps, progress_bar=True)
    rl_model.save(config.model_path)
    
    return rl_model
