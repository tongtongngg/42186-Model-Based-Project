import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
import pandas as pd
from src.data_utils import load_PL_dataset
from src.data_utils.midfield_pre import load_midfielder_data

def standardize(tensor):
    if tensor.std() == 0:
        return tensor - tensor.mean()
    return (tensor - tensor.mean()) / tensor.std()

def combined_hierarchical_model(attacker_data=None, midfielder_data=None, goalkeeper_data=None):
    # Support for specifying counts directly (useful for data generation)
    if attacker_data is None: attacker_data = {}
    if midfielder_data is None: midfielder_data = {}
    if goalkeeper_data is None: goalkeeper_data = {}
    
    n_att = attacker_data.get('n', attacker_data['dw'].shape[0] if 'dw' in attacker_data else 1)
    n_mf = midfielder_data.get('n', midfielder_data['goals'].shape[0] if 'goals' in midfielder_data else 1)
    n_gk = goalkeeper_data.get('n', goalkeeper_data['saves'].shape[0] if 'saves' in goalkeeper_data else 1)

    # --- Global Hyperpriors for Shared Parameters ---
    # ... (rest of the hyperpriors remain the same)
    # We place hyperpriors on the weights of features that appear in multiple positions
    
    # 1. Goals (Attacker, Midfielder)
    mu_w_goals = pyro.sample("mu_w_goals", dist.Normal(0.0, 1.0))
    sigma_w_goals = pyro.sample("sigma_w_goals", dist.HalfNormal(1.0))
    
    # 2. Expected Assists (Attacker, Midfielder)
    mu_w_xA = pyro.sample("mu_w_xA", dist.Normal(0.0, 1.0))
    sigma_w_xA = pyro.sample("sigma_w_xA", dist.HalfNormal(1.0))
    
    # 3. Ball Recovery (Attacker, Goalkeeper)
    mu_w_br = pyro.sample("mu_w_br", dist.Normal(0.0, 1.0))
    sigma_w_br = pyro.sample("sigma_w_br", dist.HalfNormal(1.0))
    
    # 4. Rating Baseline (All)
    mu_alpha_rating = pyro.sample("mu_alpha_rating", dist.Normal(0.0, 1.0))
    sigma_alpha_rating = pyro.sample("sigma_alpha_rating", dist.HalfNormal(1.0))
    
    # 5. Rating Noise (All)
    mu_rating_sigma = pyro.sample("mu_rating_sigma", dist.LogNormal(0.0, 0.5))

    # --- Attacker Parameters ---
    alpha_xa_att = pyro.sample("alpha_xa_att", dist.Normal(0, 1))
    beta_kp_xa_att = pyro.sample("beta_kp_xa_att", dist.Normal(0, 1))
    xa_sigma_att = pyro.sample("xa_sigma_att", dist.HalfNormal(1))
    alpha_sot_att = pyro.sample("alpha_sot_att", dist.Normal(0, 1))
    beta_ts_sot_att = pyro.sample("beta_ts_sot_att", dist.Normal(0, 1))
    sot_sigma_att = pyro.sample("sot_sigma_att", dist.HalfNormal(1))
    alpha_g_att = pyro.sample("alpha_g_att", dist.Normal(0, 1))
    beta_sot_g_att = pyro.sample("beta_sot_g_att", dist.Normal(0, 1))

    alpha_rating_att = pyro.sample("alpha_rating_att", dist.Normal(mu_alpha_rating, sigma_alpha_rating))
    w_g_att = pyro.sample("w_g_att", dist.Normal(mu_w_goals, sigma_w_goals))
    w_xa_att = pyro.sample("w_xa_att", dist.Normal(mu_w_xA, sigma_w_xA))
    w_br_att = pyro.sample("w_br_att", dist.Normal(mu_w_br, sigma_w_br))
    w_dw_att = pyro.sample("w_dw_att", dist.Normal(0, 1))
    w_sot_att = pyro.sample("w_sot_att", dist.Normal(0, 1))
    rating_sigma_att = pyro.sample("rating_sigma_att", dist.HalfNormal(mu_rating_sigma))

    # --- Midfielder Parameters ---
    # (Using logic from midfield_model.py)
    r_ohp_mf = pyro.sample("r_ohp_mf", dist.HalfNormal(10.0))
    r_shots_mf = pyro.sample("r_shots_mf", dist.HalfNormal(10.0))
    r_fouled_mf = pyro.sample("r_fouled_mf", dist.HalfNormal(10.0))
    r_goals_mf = pyro.sample("r_goals_mf", dist.HalfNormal(5.0))
    log_mu_ohp_mf = pyro.sample("log_mu_ohp_mf", dist.Normal(5.5, 1.0))
    log_mu_shots_mf = pyro.sample("log_mu_shots_mf", dist.Normal(2.0, 1.0))
    log_mu_fouled_mf = pyro.sample("log_mu_fouled_mf", dist.Normal(2.7, 1.0))
    xA_alpha_mf = pyro.sample("xA_alpha_mf", dist.HalfNormal(2.0))
    xA_beta_mf = pyro.sample("xA_beta_mf", dist.HalfNormal(1.0))
    beta_ohp_xA_mf = pyro.sample("beta_ohp_xA_mf", dist.Normal(0.0, 0.1))
    log_mu_goals_mf = pyro.sample("log_mu_goals_mf", dist.Normal(0.6, 1.0))
    beta_shots_goals_mf = pyro.sample("beta_shots_goals_mf", dist.Normal(0.0, 0.1))

    alpha_rating_mf = pyro.sample("alpha_rating_mf", dist.Normal(mu_alpha_rating, sigma_alpha_rating))
    w_ohp_mf = pyro.sample("w_ohp_mf", dist.Normal(0.0, 1.0))
    w_shots_mf = pyro.sample("w_shots_mf", dist.Normal(0.0, 1.0))
    w_fouled_mf = pyro.sample("w_fouled_mf", dist.Normal(0.0, 1.0))
    w_xA_mf = pyro.sample("w_xA_mf", dist.Normal(mu_w_xA, sigma_w_xA))
    w_goals_mf = pyro.sample("w_goals_mf", dist.Normal(mu_w_goals, sigma_w_goals))
    rating_sigma_mf = pyro.sample("rating_sigma_mf", dist.HalfNormal(mu_rating_sigma))

    # --- Goalkeeper Parameters ---
    alpha_gp_gk = pyro.sample("alpha_gp_gk", dist.Normal(0, 1))
    beta_saves_gp_gk = pyro.sample("beta_saves_gp_gk", dist.Normal(0, 1))
    gp_sigma_gk = pyro.sample("gp_sigma_gk", dist.HalfNormal(1))
    alpha_cs_gk = pyro.sample("alpha_cs_gk", dist.Normal(0, 1))
    beta_gp_cs_gk = pyro.sample("beta_gp_cs_gk", dist.Normal(0, 1))

    alpha_rating_gk = pyro.sample("alpha_rating_gk", dist.Normal(mu_alpha_rating, sigma_alpha_rating))
    w_saves_gk = pyro.sample("w_saves_gk", dist.Normal(0, 1))
    w_gp_gk = pyro.sample("w_gp_gk", dist.Normal(0, 1))
    w_cs_gk = pyro.sample("w_cs_gk", dist.Normal(0, 1))
    w_pass_gk = pyro.sample("w_pass_gk", dist.Normal(0, 1))
    w_br_gk = pyro.sample("w_br_gk", dist.Normal(mu_w_br, sigma_w_br))
    rating_sigma_gk = pyro.sample("rating_sigma_gk", dist.HalfNormal(mu_rating_sigma))

    # --- Plates ---
    
    # 1. Attacker Plate
    with pyro.plate("attacker_data", n_att):
        dw_att = pyro.sample("att_dw", dist.Normal(0, 1), obs=attacker_data.get('dw'))
        br_att = pyro.sample("att_br", dist.Normal(0, 1), obs=attacker_data.get('br'))
        kp_att = pyro.sample("att_kp", dist.Normal(0, 1), obs=attacker_data.get('kp'))
        ts_att = pyro.sample("att_ts", dist.Normal(0, 1), obs=attacker_data.get('ts'))
        
        xa_mu_att = alpha_xa_att + beta_kp_xa_att * kp_att
        xa_att = pyro.sample("att_xa", dist.Normal(xa_mu_att, xa_sigma_att), obs=attacker_data.get('xa'))
        
        sot_mu_att = alpha_sot_att + beta_ts_sot_att * ts_att
        sot_att = pyro.sample("att_sot", dist.Normal(sot_mu_att, sot_sigma_att), obs=attacker_data.get('sot'))
        
        g_log_rate_att = alpha_g_att + beta_sot_g_att * sot_att
        g_att = pyro.sample("att_g", dist.Poisson(torch.exp(g_log_rate_att)), obs=attacker_data.get('g_raw'))
        
        rating_mu_att = (alpha_rating_att + w_g_att * g_att + w_xa_att * xa_att + w_br_att * br_att + w_dw_att * dw_att + w_sot_att * sot_att)
        pyro.sample("att_rating", dist.Normal(rating_mu_att, rating_sigma_att), obs=attacker_data.get('rating'))

    # 2. Midfielder Plate
    with pyro.plate("midfielder_data", n_mf):
        ohp_mu_mf = torch.exp(log_mu_ohp_mf).expand(n_mf)
        ohp_mf = pyro.sample("mf_ohp", dist.NegativeBinomial(r_ohp_mf, r_ohp_mf / (r_ohp_mf + ohp_mu_mf)), obs=midfielder_data.get('opp_half_passes'))
        
        shots_mu_mf = torch.exp(log_mu_shots_mf).expand(n_mf)
        shots_mf = pyro.sample("mf_shots", dist.NegativeBinomial(r_shots_mf, r_shots_mf / (r_shots_mf + shots_mu_mf)), obs=midfielder_data.get('shots_outside'))
        
        fouled_mu_mf = torch.exp(log_mu_fouled_mf).expand(n_mf)
        fouled_mf = pyro.sample("mf_fouled", dist.NegativeBinomial(r_fouled_mf, r_fouled_mf / (r_fouled_mf + fouled_mu_mf)), obs=midfielder_data.get('was_fouled'))
        
        xA_concentration_mf = torch.exp(torch.clamp(torch.log(xA_alpha_mf + 1e-6) + beta_ohp_xA_mf * ohp_mf / 258.0, min=-5, max=5))
        xA_obs = pyro.sample("mf_xA", dist.Gamma(xA_concentration_mf, xA_beta_mf + 1e-4), obs=midfielder_data.get('xA'))
        
        goals_log_mu_mf = torch.clamp(log_mu_goals_mf + beta_shots_goals_mf * shots_mf / 7.5, min=-5, max=5)
        goals_mu_mf = torch.exp(goals_log_mu_mf)
        g_mf = pyro.sample("mf_g", dist.NegativeBinomial(r_ohp_mf + 1e-4, torch.clamp(r_ohp_mf / (r_ohp_mf + goals_mu_mf + 1e-4), min=1e-4, max=1-1e-4)), obs=midfielder_data.get('goals'))
        
        rating_mu_mf = (alpha_rating_mf + w_ohp_mf * torch.log1p(ohp_mf) + w_shots_mf * torch.log1p(shots_mf) + w_fouled_mf * torch.log1p(fouled_mf) + w_xA_mf * xA_obs + w_goals_mf * torch.log1p(g_mf))
        pyro.sample("mf_rating", dist.Normal(rating_mu_mf, rating_sigma_mf + 1e-4), obs=midfielder_data.get('rating'))

    # 3. Goalkeeper Plate
    with pyro.plate("goalkeeper_data", n_gk):
        saves_gk = pyro.sample("gk_saves", dist.Normal(0, 1), obs=goalkeeper_data.get('saves'))
        pass_gk = pyro.sample("gk_pass", dist.Normal(0, 1), obs=goalkeeper_data.get('accuratePasses'))
        br_gk = pyro.sample("gk_br", dist.Normal(0, 1), obs=goalkeeper_data.get('ballRecovery'))
        
        gp_mu_gk = alpha_gp_gk + beta_saves_gp_gk * saves_gk
        gp_gk = pyro.sample("gk_gp", dist.Normal(gp_mu_gk, gp_sigma_gk + 1e-4), obs=goalkeeper_data.get('goalsPrevented'))
        
        cs_log_rate_gk = torch.clamp(alpha_cs_gk + beta_gp_cs_gk * gp_gk, min=-5, max=5)
        cs_gk = pyro.sample("gk_cs", dist.Poisson(torch.exp(cs_log_rate_gk)), obs=goalkeeper_data.get('cleanSheet_raw'))
        
        rating_mu_gk = (alpha_rating_gk + w_saves_gk * saves_gk + w_gp_gk * gp_gk + w_cs_gk * cs_gk + w_pass_gk * pass_gk + w_br_gk * br_gk)
        pyro.sample("gk_rating", dist.Normal(rating_mu_gk, rating_sigma_gk + 1e-4), obs=goalkeeper_data.get('rating'))

if __name__ == "__main__":
    df = load_PL_dataset()
    
    # Prepare Attacker Data
    att_df = df[df['position'].isin(['F', 'FW', 'Attacker'])].copy()
    att_data = {
        'dw': standardize(torch.tensor(att_df['groundDuelsWon'].values, dtype=torch.float32)),
        'br': standardize(torch.tensor(att_df['ballRecovery'].values, dtype=torch.float32)),
        'kp': standardize(torch.tensor(att_df['keyPasses'].values, dtype=torch.float32)),
        'xa': standardize(torch.tensor(att_df['expectedAssists'].fillna(0).values, dtype=torch.float32)),
        'ts': standardize(torch.tensor(att_df['totalShots'].values, dtype=torch.float32)),
        'sot': standardize(torch.tensor(att_df['shotsOnTarget'].values, dtype=torch.float32)),
        'rating': standardize(torch.tensor(att_df['rating'].values, dtype=torch.float32)),
        'g_raw': torch.tensor(att_df['goals'].values, dtype=torch.float32)
    }

    # Prepare Midfielder Data
    mf_df_full = load_midfielder_data(df)
    mf_df = mf_df_full[mf_df_full['minutesPlayed'] >= 450].dropna(subset=['accurateOppositionHalfPasses', 'shotsFromOutsideTheBox', 'wasFouled', 'expectedAssists', 'goals', 'rating'])
    mf_data = {
        'opp_half_passes': torch.tensor(mf_df['accurateOppositionHalfPasses'].values, dtype=torch.float32),
        'shots_outside': torch.tensor(mf_df['shotsFromOutsideTheBox'].values, dtype=torch.float32),
        'was_fouled': torch.tensor(mf_df['wasFouled'].values, dtype=torch.float32),
        'xA': torch.tensor(mf_df['expectedAssists'].values, dtype=torch.float32),
        'goals': torch.tensor(mf_df['goals'].values, dtype=torch.float32),
        'rating': standardize(torch.tensor(mf_df['rating'].values, dtype=torch.float32))
    }

    # Prepare Goalkeeper Data
    gk_df = df[df['position'] == 'G'].copy()
    gk_data = {
        'saves': standardize(torch.tensor(gk_df['saves'].values, dtype=torch.float32)),
        'accuratePasses': standardize(torch.tensor(gk_df['accuratePasses'].values, dtype=torch.float32)),
        'ballRecovery': standardize(torch.tensor(gk_df['ballRecovery'].values, dtype=torch.float32)),
        'goalsPrevented': standardize(torch.tensor(gk_df['goalsPrevented'].fillna(0).values, dtype=torch.float32)),
        'rating': standardize(torch.tensor(gk_df['rating'].values, dtype=torch.float32)),
        'cleanSheet_raw': torch.tensor(gk_df['cleanSheet'].values, dtype=torch.float32)
    }

    print("--- Running Combined Hierarchical Model Prior Predictive Check ---")
    predictive = Predictive(combined_hierarchical_model, num_samples=1)
    # We pass None to generate prior samples without conditioning on observations
    prior_samples = predictive()

    for pos, keys in [("Attacker", ["att_g", "att_rating"]), ("Midfielder", ["mf_g", "mf_rating"]), ("Goalkeeper", ["gk_cs", "gk_rating"])]:
        print(f"\n{pos} Samples:")
        for k in keys:
            print(f"  {k}: {prior_samples[k].flatten()[:5]}")

    print("\nGlobal Hyperprior Samples:")
    for k in ["mu_w_goals", "mu_w_xA", "mu_w_br"]:
        print(f"  {k}: {prior_samples[k].item():.4f}")
