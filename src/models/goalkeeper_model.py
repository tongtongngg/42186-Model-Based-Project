import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive

from src.data_utils import load_PL_dataset
from src.data_utils.helpers import standardize


def goalkeeper_model(saves=None, accuratePasses=None, ballRecovery=None, 
                     goalsPrevented=None, cleanSheet_raw=None, rating=None):
    """
    A Probabilistic Graphical Model for Goalkeeper player Ratings.
    Inputs are standardized, EXCEPT cleanSheet_raw which uses original counts for the Poisson distribution.
    """

    # Global Parameters (Outside Plate)
    alpha_gp = pyro.sample("alpha_gp", dist.Normal(0, 1))
    beta_saves_gp = pyro.sample("beta_saves_gp", dist.Normal(0, 1))
    gp_sigma = pyro.sample("gp_sigma", dist.HalfNormal(1))

    alpha_cs = pyro.sample("alpha_cs", dist.Normal(0, 1))
    beta_gp_cs = pyro.sample("beta_gp_cs", dist.Normal(0, 1))

    alpha_rating = pyro.sample("alpha_rating", dist.Normal(0, 1))
    w_saves = pyro.sample("w_saves", dist.Normal(0, 1))
    w_gp = pyro.sample("w_gp", dist.Normal(0, 1))
    w_cs = pyro.sample("w_cs", dist.Normal(0, 1))
    w_pass = pyro.sample("w_pass", dist.Normal(0, 1))
    w_recov = pyro.sample("w_recov", dist.Normal(0, 1))
    rating_sigma = pyro.sample("rating_sigma", dist.HalfNormal(1))

    n_obs = 1
    if saves is not None:
        n_obs = saves.shape[0]
    elif rating is not None:
        n_obs = rating.shape[0]

    # plate to indicate conditionally independent observations
    with pyro.plate("data", n_obs):
        # Standardized priors centered around 0
        saves_obs = pyro.sample("saves", dist.Normal(0, 1), obs=saves)
        accuratePasses_obs = pyro.sample("accuratePasses", dist.Normal(0, 1), obs=accuratePasses)
        ballRecovery_obs = pyro.sample("ballRecovery", dist.Normal(0, 1), obs=ballRecovery)

        # goalsPrevented (continuous, standardized) depends on saves.
        gp_mu = alpha_gp + beta_saves_gp * saves_obs
        goalsPrevented_obs = pyro.sample("goalsPrevented", dist.Normal(gp_mu, gp_sigma), obs=goalsPrevented)

        # cleanSheet (discrete counts) depends on goalsPrevented.
        cs_log_rate = alpha_cs + beta_gp_cs * goalsPrevented_obs
        cleanSheet_obs = pyro.sample("cleanSheet", dist.Poisson(torch.exp(cs_log_rate)), obs=cleanSheet_raw)

        # Rating depends on all 5 nodes. 
        rating_mu = (alpha_rating + 
                     w_saves * saves_obs + 
                     w_gp * goalsPrevented_obs + 
                     w_cs * cleanSheet_obs + 
                     w_pass * accuratePasses_obs + 
                     w_recov * ballRecovery_obs)
        
        rating_obs = pyro.sample("rating", dist.Normal(rating_mu, rating_sigma), obs=rating)

    return rating_obs

if __name__ == "__main__":
    df = load_PL_dataset()
    gk_df = df[df['position'] == 'G'].copy()
    
    gk_df['goalsPrevented'] = gk_df['goalsPrevented'].fillna(gk_df['goalsPrevented'].mean())

    data = {
        'saves': torch.tensor(gk_df['saves'].values, dtype=torch.float32),
        'accuratePasses': torch.tensor(gk_df['accuratePasses'].values, dtype=torch.float32),
        'ballRecovery': torch.tensor(gk_df['ballRecovery'].values, dtype=torch.float32),
        'goalsPrevented': torch.tensor(gk_df['goalsPrevented'].values, dtype=torch.float32),
        'rating': torch.tensor(gk_df['rating'].values, dtype=torch.float32)
    }
    
    # Keep cleanSheet raw (not standardized) for the Poisson distribution
    cleanSheet_raw = torch.tensor(gk_df['cleanSheet'].values, dtype=torch.float32)

    std_data = {k: standardize(v) for k, v in data.items()}
    std_data['cleanSheet_raw'] = cleanSheet_raw

    predictive = Predictive(goalkeeper_model, num_samples=1)
    prior_samples = predictive()


    print("Generated 1 sample of fake data from the priors:")
    for k in ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented', 'rating']:
        print(f"Sampled {k}: {prior_samples[k].flatten()[:5]}") 
