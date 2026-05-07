import torch
from enum import Enum
from src.models.parameter_recovery import run_parameter_recovery
from src.models.goalkeeper_model import goalkeeper_model
from src.models.attacker_model import forward_model
from src.models.defender_model import defender_model
from src.models.midfield_model import midfielder_model

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
        
    elif model_type == ModelType.MIDFIELDER:
        model = midfielder_model

        true_params = {
            # NegBin dispersion
            'r_ohp':    torch.tensor(5.0),
            'r_shots':  torch.tensor(3.0),
            'r_fouled': torch.tensor(3.0),
            'r_goals':  torch.tensor(2.0),

            'log_mu_ohp':    torch.tensor(5.5),
            'log_mu_shots':  torch.tensor(2.0),
            'log_mu_fouled': torch.tensor(2.7),

            # expectedAssists (log1p scale)
            'xA_mu_base': torch.tensor(0.3),
            'xA_sigma':   torch.tensor(0.5),
            'beta_ohp_xA': torch.tensor(0.08),

            # goals log-mean and shots dependency
            'log_mu_goals':     torch.tensor(0.6),
            'beta_shots_goals': torch.tensor(0.2),

            # rating regression
            'alpha_rating': torch.tensor(0.0),
            'w_ohp':        torch.tensor(0.45),
            'w_shots':      torch.tensor(0.23),
            'w_fouled':     torch.tensor(0.18),
            'w_xA':         torch.tensor(0.37),
            'w_goals':      torch.tensor(0.14),
            'rating_sigma': torch.tensor(0.3),
        }

        obs_mapping = {
            'opp_half_passes': 'opp_half_passes',
            'shots_outside':   'shots_outside',
            'was_fouled':      'was_fouled',
            'xA':              'xA',
            'goals':           'goals',
            'rating':          'rating',
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
    test_model_recovery(ModelType.MIDFIELDER, num_samples=2000, num_steps=3000)
