import torch
from enum import Enum
from src.models.defender_model import defender_model
from src.models.parameter_recovery import run_parameter_recovery
from src.models.goalkeeper_model import goalkeeper_model
from src.models.attacker_model import forward_model

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
        
        obs_mapping = {
            'saves': 'saves',
            'accuratePasses': 'accuratePasses',
            'ballRecovery': 'ballRecovery',
            'goalsPrevented': 'goalsPrevented',
            'cleanSheet_raw': 'cleanSheet',
            'rating': 'rating'
        }

    elif model_type == ModelType.FORWARD: 
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
        
        obs_mapping = {
            'dw': 'dw',
            'br': 'br',
            'kp': 'kp',
            'xa': 'xa',
            'ts': 'ts',
            'sot': 'sot',
            'g_raw': 'g',  
            'rating': 'rating'
        }
        
    elif model_type == ModelType.DEFENDER:
        model = defender_model

        true_params = {
            # touches
            "touches_mu": torch.tensor(0.0),
            "touches_sigma": torch.tensor(1.0),

            # totalDuelsWon
            "totalDuelsWon_mu": torch.tensor(0.0),
            "totalDuelsWon_sigma": torch.tensor(1.0),

            # accuratePasses <- touches
            "alpha_passes": torch.tensor(0.0),
            "w_touches_to_passes": torch.tensor(0.7),
            "passes_sigma": torch.tensor(0.4),

            # clearances <- totalDuelsWon
            "alpha_clearances": torch.tensor(0.0),
            "w_duels_to_clearances": torch.tensor(0.6),
            "clearances_sigma": torch.tensor(0.4),

            # cleanSheet <- clearances
            "alpha_cleanSheet": torch.tensor(-0.5),
            "w_clearances_to_cleanSheet": torch.tensor(0.8),

            # rating <- touches + accuratePasses + cleanSheet
            "alpha_rating": torch.tensor(0.0),
            "w_touches_to_rating": torch.tensor(0.3),
            "w_passes_to_rating": torch.tensor(0.5),
            "w_cleanSheet_to_rating": torch.tensor(0.4),
            "rating_sigma": torch.tensor(0.3),
        }

        obs_mapping = {
            "touches": "touches",
            "accuratePasses": "accuratePasses",
            "totalDuelsWon": "totalDuelsWon",
            "clearances": "clearances",
            "cleanSheet": "cleanSheet",
            "rating": "rating",
        }

    run_parameter_recovery(
        model=model,
        true_params=true_params,
        obs_mapping=obs_mapping,
        num_samples=num_samples,
        num_steps=num_steps,
        lr=lr
    )

if __name__ == "__main__":
    test_model_recovery(ModelType.DEFENDER, num_samples=2000, num_steps=3000)
