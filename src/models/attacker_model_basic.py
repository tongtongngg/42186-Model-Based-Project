import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
import pandas as pd
from src.data_utils import load_PL_dataset

def standardize(tensor):
    """Standardizes a tensor to have mean 0 and standard deviation 1."""
    return (tensor - tensor.mean()) / tensor.std()

def forward_model(dw=None, br=None, kp=None, xa=None, ts=None, sot=None, g_raw=None, rating=None):
    """
    A Probabilistic Graphical Model for Forward (Attacker) player Ratings.
    Inputs are standardized, EXCEPT g_raw (goals) which uses original counts for the Poisson distribution.
    """

    # Global Parameters (Outside Plate)
    
    # Creativity Hierarchy
    alpha_xa = pyro.sample("alpha_xa", dist.Normal(0, 1))
    beta_kp_xa = pyro.sample("beta_kp_xa", dist.Normal(0, 1))
    xa_sigma = pyro.sample("xa_sigma", dist.HalfNormal(1))

    # Goal Threat Hierarchy
    alpha_sot = pyro.sample("alpha_sot", dist.Normal(0, 1))
    beta_ts_sot = pyro.sample("beta_ts_sot", dist.Normal(0, 1))
    sot_sigma = pyro.sample("sot_sigma", dist.HalfNormal(1))

    alpha_g = pyro.sample("alpha_g", dist.Normal(0, 1))
    beta_sot_g = pyro.sample("beta_sot_g", dist.Normal(0, 1))

    # Rating Target
    alpha_rating = pyro.sample("alpha_rating", dist.Normal(0, 1))
    w_g = pyro.sample("w_g", dist.Normal(0, 1))
    w_xa = pyro.sample("w_xa", dist.Normal(0, 1))
    w_dw = pyro.sample("w_dw", dist.Normal(0, 1))
    w_br = pyro.sample("w_br", dist.Normal(0, 1))
    w_sot = pyro.sample("w_sot", dist.Normal(0, 1))
    rating_sigma = pyro.sample("rating_sigma", dist.HalfNormal(1))

    # Determine batch size
    n_obs = 1
    if dw is not None:
        n_obs = dw.shape[0]
    elif rating is not None:
        n_obs = rating.shape[0]

    # Use a plate to indicate conditionally independent observations
    with pyro.plate("data", n_obs):
        # 1. Independent Root Nodes (Standardized priors centered around 0)
        dw_obs = pyro.sample("dw", dist.Normal(0, 1), obs=dw)
        br_obs = pyro.sample("br", dist.Normal(0, 1), obs=br)
        kp_obs = pyro.sample("kp", dist.Normal(0, 1), obs=kp)
        ts_obs = pyro.sample("ts", dist.Normal(0, 1), obs=ts)

        # 2. Creativity Hierarchy
        xa_mu = alpha_xa + beta_kp_xa * kp_obs
        xa_obs = pyro.sample("xa", dist.Normal(xa_mu, xa_sigma), obs=xa)

        # 3. Goal Threat Hierarchy
        sot_mu = alpha_sot + beta_ts_sot * ts_obs
        sot_obs = pyro.sample("sot", dist.Normal(sot_mu, sot_sigma), obs=sot)

        # goals (discrete counts) depends on shotsOnTarget.
        g_log_rate = alpha_g + beta_sot_g * sot_obs
        g_obs = pyro.sample("g", dist.Poisson(torch.exp(g_log_rate)), obs=g_raw)

        # 4. Final Rating Combination
        rating_mu = (alpha_rating + 
                     w_g * g_obs + 
                     w_xa * xa_obs + 
                     w_dw * dw_obs + 
                     w_br * br_obs + 
                     w_sot * sot_obs)
        
        rating_obs = pyro.sample("rating", dist.Normal(rating_mu, rating_sigma), obs=rating)

    return rating_obs

if __name__ == "__main__":
    df = load_PL_dataset()
    # Assuming 'F' or 'FW' is the label for Forwards in your dataset
    fwd_df = df[df['position'].isin(['F', 'FW', 'Attacker'])].copy()
    
    # Handle potential nulls just in case
    fwd_df['expectedAssists'] = fwd_df['expectedAssists'].fillna(0)

    data = {
        'dw': torch.tensor(fwd_df['groundDuelsWon'].values, dtype=torch.float32),
        'br': torch.tensor(fwd_df['ballRecovery'].values, dtype=torch.float32),
        'kp': torch.tensor(fwd_df['keyPasses'].values, dtype=torch.float32),
        'xa': torch.tensor(fwd_df['expectedAssists'].values, dtype=torch.float32),
        'ts': torch.tensor(fwd_df['totalShots'].values, dtype=torch.float32),
        'sot': torch.tensor(fwd_df['shotsOnTarget'].values, dtype=torch.float32),
        'rating': torch.tensor(fwd_df['rating'].values, dtype=torch.float32)
    }
    
    # Keep goals raw (not standardized) for the Poisson distribution
    g_raw = torch.tensor(fwd_df['goals'].values, dtype=torch.float32)

    # Standardize Continuous Data
    std_data = {k: standardize(v) for k, v in data.items()}
    std_data['g_raw'] = g_raw

    print("--- Ancestral Sampling (Prior Predictive Checks) ---")
    predictive = Predictive(forward_model, num_samples=1)
    prior_samples = predictive()

    print("Generated 1 sample of fake data from the DAG priors:")
    for k in ['dw', 'xa', 'sot', 'g', 'rating']:
        print(f"Sampled {k}: {prior_samples[k].flatten()[:5]}...") 

    print("\nThe hybrid continuous/discrete PGM for Forwards is setup and ready for inference!")