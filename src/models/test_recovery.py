import torch
from enum import Enum
from src.models.parameter_recovery import run_parameter_recovery
from src.models.goalkeeper_model import goalkeeper_model
from src.models.attacker_model_basic import forward_model
# Import other models here as they are created
# from src.models.defender_model import defender_model

class ModelType(Enum):
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"
    MIDFIELDER = "midfielder"
    FORWARD = "forward"

def test_model_recovery(model_type: ModelType, num_samples=2000, num_steps=2000, lr=0.05):
    """
    Sets up the true parameters and mapping for the specified model type
    and runs the parameter recovery test.
    """
    print(f"--- Running Parameter Recovery for {model_type.name} ---")

    if model_type == ModelType.GOALKEEPER:
        model = goalkeeper_model
        
        # Define sensible "True" parameters to bury in the fake data
        # Weights (can be negative or positive)
        # Sigmas (must be > 0)
        true_params = {
            'alpha_gp': torch.tensor(0.0),
            'beta_saves_gp': torch.tensor(0.6),
            'gp_sigma': torch.tensor(0.5),
            
            'alpha_cs': torch.tensor(0.0),
            'beta_gp_cs': torch.tensor(0.4),
            
            'alpha_rating': torch.tensor(0.0),
            'w_saves': torch.tensor(0.3),
            'w_gp': torch.tensor(0.4),
            'w_cs': torch.tensor(0.1),
            'w_pass': torch.tensor(0.2),
            'w_recov': torch.tensor(0.1),
            'rating_sigma': torch.tensor(0.2)
        }
        
        # Map the model's kwargs to the generated sample site names
        obs_mapping = {
            'saves': 'saves',
            'accuratePasses': 'accuratePasses',
            'ballRecovery': 'ballRecovery',
            'goalsPrevented': 'goalsPrevented',
            'cleanSheet_raw': 'cleanSheet',
            'rating': 'rating'
        }

    elif model_type == ModelType.FORWARD:  # <--- Added Forward Logic
        model = forward_model
        
        true_params = {
            'alpha_xa': torch.tensor(0.0),
            'beta_kp_xa': torch.tensor(0.8),
            'xa_sigma': torch.tensor(0.3),
            
            'alpha_sot': torch.tensor(0.0),
            'beta_ts_sot': torch.tensor(0.7),
            'sot_sigma': torch.tensor(0.4),
            
            'alpha_g': torch.tensor(-1.0), # Negative intercept since goals are rare
            'beta_sot_g': torch.tensor(0.5),
            
            'alpha_rating': torch.tensor(0.0),
            'w_g': torch.tensor(1.2),
            'w_xa': torch.tensor(0.9),
            'w_dw': torch.tensor(0.4),
            'w_br': torch.tensor(0.2),
            'w_sot': torch.tensor(0.3),
            'rating_sigma': torch.tensor(0.3)
        }
        
        # Map kwarg names -> Sample Site names
        obs_mapping = {
            'dw': 'dw',
            'br': 'br',
            'kp': 'kp',
            'xa': 'xa',
            'ts': 'ts',
            'sot': 'sot',
            'g_raw': 'g',  # Maps 'g_raw' argument to the 'g' Poisson site
            'rating': 'rating'
        }
        
    elif model_type == ModelType.DEFENDER:
        # Placeholder for defender setup
        print("Defender model not yet implemented for recovery test.")
        return
    else:
        print(f"{model_type.name} not yet configured.")
        return

    # 3. Run the recovery!
    run_parameter_recovery(
        model=model,
        true_params=true_params,
        obs_mapping=obs_mapping,
        num_samples=num_samples,
        num_steps=num_steps,
        lr=lr
    )

if __name__ == "__main__":
    # Test the Goalkeeper model
    test_model_recovery(ModelType.FORWARD, num_samples=2000, num_steps=3000)
