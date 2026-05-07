import torch
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from src.models.combined_hierarchical_model import combined_hierarchical_model

def test_combined_recovery(num_samples_per_pos=1000, num_steps=3000, lr=0.02):
    """
    Parameter recovery test for the Combined Hierarchical Model.
    """
    pyro.clear_param_store()
    
    true_params = {
        # global Hyperpriors
        "mu_w_goals": torch.tensor(0.5),
        "sigma_w_goals": torch.tensor(0.2),
        "mu_w_xA": torch.tensor(0.8),
        "sigma_w_xA": torch.tensor(0.1),
        "mu_w_br": torch.tensor(0.3),
        "sigma_w_br": torch.tensor(0.2),
        "mu_alpha_rating": torch.tensor(0.0),
        "sigma_alpha_rating": torch.tensor(0.5),
        "mu_rating_sigma": torch.tensor(0.2),
        
        # position-Specific Parameters 
        "w_dw_att": torch.tensor(0.4),
        "w_sot_att": torch.tensor(0.3),
        "w_ohp_mf": torch.tensor(0.6),
        "w_shots_mf": torch.tensor(0.4),
        "w_fouled_mf": torch.tensor(0.2),
        "w_saves_gk": torch.tensor(0.8),
        "w_gp_gk": torch.tensor(0.5),
        "w_cs_gk": torch.tensor(0.4),
        "w_pass_gk": torch.tensor(0.3),
    }
    
    print(f"Generating synthetic data ({num_samples_per_pos} samples per position)...")
    conditioned_model = pyro.condition(combined_hierarchical_model, data=true_params)
    
    dummy_data = {
        'attacker_data': {'n': num_samples_per_pos},
        'midfielder_data': {'n': num_samples_per_pos},
        'goalkeeper_data': {'n': num_samples_per_pos}
    }
    
    # Generate ONE trace with the full plate size
    fake_data = Predictive(conditioned_model, num_samples=1)(**dummy_data)
    
    print("\nSynthetic Data Stats:")
    for pos_prefix, pos_name in [("att", "Attacker"), ("mf", "Midfielder"), ("gk", "Goalkeeper")]:
        print(f"  {pos_name}:")
        for k in fake_data.keys():
            if k.startswith(pos_prefix):
                val = fake_data[k].detach()
                print(f"    {k:<15}: mean={val.mean():.2f}, std={val.std():.2f}, min={val.min():.2f}, max={val.max():.2f}")

    # 3. Prepare kwargs for SVI
    # Map fake_data (site names) back to the kwargs the model expects
    svi_kwargs = {
        'attacker_data': {
            'dw': fake_data['att_dw'].squeeze(),
            'br': fake_data['att_br'].squeeze(),
            'kp': fake_data['att_kp'].squeeze(),
            'xa': fake_data['att_xa'].squeeze(),
            'ts': fake_data['att_ts'].squeeze(),
            'sot': fake_data['att_sot'].squeeze(),
            'rating': fake_data['att_rating'].squeeze(),
            'g_raw': fake_data['att_g'].squeeze()
        },
        'midfielder_data': {
            'opp_half_passes': fake_data['mf_ohp'].squeeze(),
            'shots_outside': fake_data['mf_shots'].squeeze(),
            'was_fouled': fake_data['mf_fouled'].squeeze(),
            'xA': fake_data['mf_xA'].squeeze(),
            'goals': fake_data['mf_g'].squeeze(),
            'rating': fake_data['mf_rating'].squeeze()
        },
        'goalkeeper_data': {
            'saves': fake_data['gk_saves'].squeeze(),
            'accuratePasses': fake_data['gk_pass'].squeeze(),
            'ballRecovery': fake_data['gk_br'].squeeze(),
            'goalsPrevented': fake_data['gk_gp'].squeeze(),
            'rating': fake_data['gk_rating'].squeeze(),
            'cleanSheet_raw': fake_data['gk_cs'].squeeze()
        }
    }
    
    # 4. Run SVI
    print(f"Running SVI for {num_steps} steps...")
    guide = AutoNormal(combined_hierarchical_model)
    optimizer = Adam({"lr": lr})
    svi = SVI(combined_hierarchical_model, guide, optimizer, loss=Trace_ELBO())
    
    for step in range(num_steps):
        loss = svi.step(**svi_kwargs)
        if step % 500 == 0:
            print(f"Step {step:>4} | Loss: {loss:.2f}")
            
    # 5. Results
    print("\n" + "="*55)
    print("        COMBINED MODEL PARAMETER RECOVERY RESULTS")
    print("="*55)
    print(f"{'Parameter':<25} | {'True':>7} | {'Inferred':>8} | {'Error':>6}")
    print("-" * 55)
    
    inferred_medians = guide.median()
    for param_name, true_tensor in true_params.items():
        true_val = true_tensor.item()
        if param_name in inferred_medians:
            inferred_val = inferred_medians[param_name].item()
            error = abs(true_val - inferred_val)
            print(f"{param_name:<25} | {true_val:>7.3f} | {inferred_val:>8.3f} | {error:>6.3f}")
        else:
            print(f"{param_name:<25} | {true_val:>7.3f} | NOT FOUND")
    print("="*55)

if __name__ == "__main__":
    test_combined_recovery(num_samples_per_pos=2000, num_steps=3000)
