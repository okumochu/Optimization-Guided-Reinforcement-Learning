import os

class Config:
    def __init__(self):
        
        # path
        self.data_file = "offline_inventory_data.csv"
        self.model_path = os.path.join("src", "model", "ppo_inventory_model.pth")
        self.training_log_dir = os.path.join("src", "model", "training_log")
        
        # rlero hyper-parameters
        self.entropy_threshold = 0.3 # threshold for RLeRO
        self.delta_D = 5.0 # demand uncertainty for robust optimization
        self.delta_L = 0.5 # lead time uncertainty for robust optimization
        
        # problem formulation
        self.C_h = 1.0 # holding cost
        self.C_s = 5.0 # shortage cost
        self.C_o = 2.0 # ordering cost
        self.max_order = 100
        self.max_inventory = 100
        self.max_demand = 100
        
        # training
        self.n_steps = 5000000 # number of training steps
        self.training_period = 100000 # number of training periods
        self.test_period = 1000 # number of test periods
