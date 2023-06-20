from base import StatisticFunctions as sf


class Config:
    """
    Before training a model, you should provide your configs
    """
    def __init__(self):
        self.main_config = None
        self.window = 100
        self.step = 100  # this will be used for rolling window stride size
    
    def create_main_config(self):

        self.main_config = {
    
            sf.mean_func: {'AccV': [self.window]},
    
            sf.median_func: {'AccAP': [self.window]},
    
            sf.abs_energy_func: {'AccML': [self.window]},
    
            sf.std_func: {'AccAP': [self.window]},
    
            sf.var_func: {'AccV': [self.window]},
    
            sf.min_func: {'AccAP': [self.window]},
    
            sf.max_func: {'AccV': [self.window]},
    
            sf.skew_func: {'AccML': [self.window]},
    
            sf.kurt_func: {'AccAP': [self.window]},
    
            sf.mse_func: {'AccV': [self.window]},
    
            sf.mnx_func: {'AccML': [self.window]},
    
            sf.mean_abs_func: {'AccAP': [self.window]},
    
            sf.slope_sign_func: {'AccML': [self.window]},
    
            sf.waveform_length_func: {'AccAP': [self.window]},
    
            sf.integrated_emg_func: {'AccV': [self.window]},

        }

