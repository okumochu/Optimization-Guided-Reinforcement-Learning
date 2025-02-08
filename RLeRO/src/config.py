import os

class Config:
    def __init__(self):
        
        # path
        self.data_file = "offline_inventory_data.csv"
        self.model_path = os.path.join("src", "model", "ppo_inventory_model.pth")
        self.training_log_dir = os.path.join("src", "model", "training_log")
        
        # rlero hyper-parameters
        self.entropy_threshold = 0.8
        self.delta_D = 5.0
        self.delta_L = 0.5
        
        # problem formulation
        self.C_h = 1.0
        self.C_s = 5.0
        self.C_o = 2.0
        self.max_order = 100
        self.max_inventory = 100
        self.max_demand = 100
        
        # training
        self.n_steps = 50000
