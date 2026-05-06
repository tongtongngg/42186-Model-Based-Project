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

def combined_hierarchical_model(attacker_data=None, midfielder_data=None, goalkeeper_data=None, defender_data=None):
    if attacker_data is None: attacker_data = {}
    if midfielder_data is None: midfielder_data = {}
    if goalkeeper_data is None: goalkeeper_data = {}
    if defender_data is None: defender_data = {}
    
    n_att = attacker_data.get('n', attacker_data['dw'].shape[0] if 'dw' in attacker_data else 1)
    n_mf = midfielder_data.get('n', midfielder_data['goals'].shape[0] if 'goals' in midfielder_data else 1)
    n_gk = goalkeeper_data.get('n', goalkeeper_data['saves'].shape[0] if 'saves' in goalkeeper_data else 1)
    n_def = defender_data.get('n', defender_data['touches'].shape[0] if 'touches' in defender_data else 1)

    mu_w_goals = pyro.sample("mu_w_goals", dist.Normal(0.0, 1.0))
    sigma_w_goals = pyro.sample("sigma_w_goals", dist.HalfNormal(1.0))
    mu_w_xA = pyro.sample("mu_w_xA", dist.Normal(0.0, 1.0))
    sigma_w_xA = pyro.sample("sigma_w_xA", dist.HalfNormal(1.0))
    mu_w_br = pyro.sample("mu_w_br", dist.Normal(0.0, 1.0))
    sigma_w_br = pyro.sample("sigma_w_br", dist.HalfNormal(1.0))
    mu_alpha_rating = pyro.sample("mu_alpha_rating", dist.Normal(0.0, 1.0))
    sigma_alpha_rating = pyro.sample("sigma_alpha_rating", dist.HalfNormal(1.0))
    mu_rating_sigma = pyro.sample("mu_rating_sigma", dist.LogNormal(0.0, 0.5))

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

    r_ohp_mf = pyro.sample("r_ohp_mf", dist.HalfNormal(10.0))
    r_shots_mf = pyro.sample("r_shots_mf", dist.HalfNormal(10.0))
    r_fouled_mf = pyro.sample("r_fouled_mf", dist.HalfNormal(10.0))
    r_goals_mf = pyro.sample("r_goals_mf", dist.HalfNormal(5.0))
    log_mu_ohp_mf = pyro.sample("log_mu_ohp_mf", dist.Normal(5.5, 1.0))
    log_mu_shots_mf = pyro.sample("log_mu_shots_mf", dist.Normal(2.0, 1.0))
    log_mu_fouled_mf = pyro.sample("log_mu_fouled_mf", dist.Normal(2.7, 1.0))
    xA_mu_base_mf = pyro.sample("xA_mu_base_mf", dist.Normal(0.0, 1.0))
    xA_sigma_mf = pyro.sample("xA_sigma_mf", dist.HalfNormal(1.0))
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

    touches_mu_def = pyro.sample("touches_mu_def", dist.Normal(0, 1))
    touches_sigma_def = pyro.sample("touches_sigma_def", dist.HalfNormal(1))
    totalDuelsWon_mu_def = pyro.sample("totalDuelsWon_mu_def", dist.Normal(0, 1))
    totalDuelsWon_sigma_def = pyro.sample("totalDuelsWon_sigma_def", dist.HalfNormal(1))
    alpha_passes_def = pyro.sample("alpha_passes_def", dist.Normal(0, 1))
    w_touches_to_passes_def = pyro.sample("w_touches_to_passes_def", dist.Normal(0, 1))
    passes_sigma_def = pyro.sample("passes_sigma_def", dist.HalfNormal(1))
    alpha_clearances_def = pyro.sample("alpha_clearances_def", dist.Normal(0, 1))
    w_duels_to_clearances_def = pyro.sample("w_duels_to_clearances_def", dist.Normal(0, 1))
    clearances_sigma_def = pyro.sample("clearances_sigma_def", dist.HalfNormal(1))
    alpha_cs_def = pyro.sample("alpha_cs_def", dist.Normal(0, 1))
    w_clearances_to_cs_def = pyro.sample("w_clearances_to_cs_def", dist.Normal(0, 1))
    alpha_rating_def = pyro.sample("alpha_rating_def", dist.Normal(mu_alpha_rating, sigma_alpha_rating))
    w_touches_def = pyro.sample("w_touches_def", dist.Normal(0, 1))
    w_passes_def = pyro.sample("w_passes_def", dist.Normal(0, 1))
    w_cs_def = pyro.sample("w_cs_def", dist.Normal(0, 1))
    rating_sigma_def = pyro.sample("rating_sigma_def", dist.HalfNormal(mu_rating_sigma))

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

    with pyro.plate("midfielder_data", n_mf):
        ohp_mu_mf = torch.exp(log_mu_ohp_mf).expand(n_mf)
        ohp_mf = pyro.sample("mf_ohp", dist.NegativeBinomial(r_ohp_mf, ohp_mu_mf / (r_ohp_mf + ohp_mu_mf)), obs=midfielder_data.get('opp_half_passes'))
        shots_mu_mf = torch.exp(log_mu_shots_mf).expand(n_mf)
        shots_mf = pyro.sample("mf_shots", dist.NegativeBinomial(r_shots_mf, shots_mu_mf / (r_shots_mf + shots_mu_mf)), obs=midfielder_data.get('shots_outside'))
        fouled_mu_mf = torch.exp(log_mu_fouled_mf).expand(n_mf)
        fouled_mf = pyro.sample("mf_fouled", dist.NegativeBinomial(r_fouled_mf, fouled_mu_mf / (r_fouled_mf + fouled_mu_mf)), obs=midfielder_data.get('was_fouled'))
        xA_mu_val = xA_mu_base_mf + beta_ohp_xA_mf * ohp_mf / 258.0
        xA_obs = pyro.sample("mf_xA", dist.Normal(xA_mu_val, xA_sigma_mf + 1e-4), obs=midfielder_data.get('xA'))
        goals_log_mu_mf = log_mu_goals_mf + beta_shots_goals_mf * shots_mf / 7.5
        goals_mu_mf = torch.exp(goals_log_mu_mf)
        g_mf = pyro.sample("mf_g", dist.NegativeBinomial(r_goals_mf, goals_mu_mf / (r_goals_mf + goals_mu_mf)), obs=midfielder_data.get('goals'))
        rating_mu_mf = (alpha_rating_mf + w_ohp_mf * torch.log1p(ohp_mf) + w_shots_mf * torch.log1p(shots_mf) + w_fouled_mf * torch.log1p(fouled_mf) + w_xA_mf * xA_obs + w_goals_mf * torch.log1p(g_mf))
        pyro.sample("mf_rating", dist.Normal(rating_mu_mf, rating_sigma_mf + 1e-4), obs=midfielder_data.get('rating'))

    with pyro.plate("goalkeeper_data", n_gk):
        saves_gk = pyro.sample("gk_saves", dist.Normal(0, 1), obs=goalkeeper_data.get('saves'))
        pass_gk = pyro.sample("gk_pass", dist.Normal(0, 1), obs=goalkeeper_data.get('accuratePasses'))
        br_gk = pyro.sample("gk_br", dist.Normal(0, 1), obs=goalkeeper_data.get('ballRecovery'))
        gp_mu_gk = alpha_gp_gk + beta_saves_gp_gk * saves_gk
        gp_gk = pyro.sample("gk_gp", dist.Normal(gp_mu_gk, gp_sigma_gk + 1e-4), obs=goalkeeper_data.get('goalsPrevented'))
        cs_log_rate_gk = alpha_cs_gk + beta_gp_cs_gk * gp_gk
        cs_gk = pyro.sample("gk_cs", dist.Poisson(torch.exp(cs_log_rate_gk)), obs=goalkeeper_data.get('cleanSheet_raw'))
        rating_mu_gk = (alpha_rating_gk + w_saves_gk * saves_gk + w_gp_gk * gp_gk + w_cs_gk * cs_gk + w_pass_gk * pass_gk + w_br_gk * br_gk)
        pyro.sample("gk_rating", dist.Normal(rating_mu_gk, rating_sigma_gk + 1e-4), obs=goalkeeper_data.get('rating'))

    with pyro.plate("defender_data", n_def):
        touches_def = pyro.sample("def_touches", dist.Normal(touches_mu_def, touches_sigma_def), obs=defender_data.get('touches'))
        duels_def = pyro.sample("def_duels", dist.Normal(totalDuelsWon_mu_def, totalDuelsWon_sigma_def), obs=defender_data.get('totalDuelsWon'))
        passes_mu_def = alpha_passes_def + w_touches_to_passes_def * touches_def
        passes_def = pyro.sample("def_passes", dist.Normal(passes_mu_def, passes_sigma_def), obs=defender_data.get('accuratePasses'))
        clearances_mu_def = alpha_clearances_def + w_duels_to_clearances_def * duels_def
        clearances_def = pyro.sample("def_clearances", dist.Normal(clearances_mu_def, clearances_sigma_def), obs=defender_data.get('clearances'))
        cs_rate_def = torch.exp(alpha_cs_def + w_clearances_to_cs_def * clearances_def)
        cs_def = pyro.sample("def_cs", dist.Poisson(cs_rate_def), obs=defender_data.get('cleanSheet'))
        rating_mu_def = (alpha_rating_def + w_touches_def * touches_def + w_passes_def * passes_def + w_cs_def * cs_def)
        pyro.sample("def_rating", dist.Normal(rating_mu_def, rating_sigma_def + 1e-4), obs=defender_data.get('rating'))
