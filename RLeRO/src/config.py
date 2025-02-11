import os

class Config:
    def __init__(self):
        
        # path
        self.data_file = "offline_inventory_data.csv"
        self.model_path = os.path.join("src", "model", "ppo_inventory_model.pth")
        self.training_log_dir = os.path.join("src", "model", "training_log")
        
        # rlero hyper-parameters
        self.entropy_threshold = 0.1 # threshold for RLeRO
        self.delta_D = 5.0 # demand uncertainty for robust optimization
        self.delta_Y = 0.1 # yield rate uncertainty (±10%)
        
        # problem formulation
        self.C_h = 1.0 # holding cost
        self.C_s = 5.0 # shortage cost
        self.C_o = 2.0 # production cost per unit
        
        self.max_demand = 100
        self.max_order = 100
        self.max_inventory = 100
        
        # demand
        self.base_demand = 50        
        self.train_demand_volatility = 0.1
        self.test_demand_volatility = 0.2
        
        # yield rate
        self.base_yield_rate = 0.9
        self.max_yield_rate = 1.0
        self.min_yield_rate = 0.5
        self.train_yield_rate_volatility = 0.0025
        self.test_yield_rate_volatility = 0.01
        
        # training
        self.training_period = 10000 # number of training periods
        self.test_period = 1000 # number of test periods
        self.production_level_option = [0, 20, 40, 60, 80, 100]
        self.n_steps = self.training_period * 1000 # number of training steps

