import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
import pandas as pd
from src.data_utils import load_PL_dataset
from src.data_utils.midfield_pre import load_midfielder_data

FEATURES = [
    'accurateOppositionHalfPasses',
    'shotsFromOutsideTheBox',
    'wasFouled',
    'expectedAssists',
    'goals',
]
MIN_MINUTES = 450

# Approximate observed log-means used to centre priors (from midfield_analysis.py output)
_LOG_MU_OHP    = 5.5   # log(258)
_LOG_MU_SHOTS  = 2.0   # log(7.5)
_LOG_MU_FOULED = 2.7   # log(15)
_LOG_MU_GOALS  = 0.6   # log(1.8)
_MEAN_OHP      = 258.0
_MEAN_SHOTS    = 7.5


def standardize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.mean()) / tensor.std()


def midfielder_model(
    opp_half_passes=None,
    shots_outside=None,
    was_fouled=None,
    xA=None,
    goals=None,
    rating=None,
):
    """
    Bayesian PGM for midfielder ratings.

    DAG:
        accurateOppositionHalfPasses (NegBin) --> expectedAssists (Gamma)
        shotsFromOutsideTheBox       (NegBin) --> goals            (NegBin)
        wasFouled                    (NegBin)
        all five nodes --> rating (Normal)

    Count features use Negative Binomial (overdispersed counts confirmed in analysis).
    expectedAssists uses Gamma (continuous, right-skewed).
    Rating regression uses log1p of counts to put all inputs on a comparable scale.
    """

    # --- Dispersion parameters (one per NegBin feature) ---
    r_ohp    = pyro.sample("r_ohp",    dist.HalfNormal(10.0))
    r_shots  = pyro.sample("r_shots",  dist.HalfNormal(10.0))
    r_fouled = pyro.sample("r_fouled", dist.HalfNormal(10.0))
    r_goals  = pyro.sample("r_goals",  dist.HalfNormal(5.0))

    # --- Global log-means for root NegBin nodes ---
    log_mu_ohp    = pyro.sample("log_mu_ohp",    dist.Normal(_LOG_MU_OHP,    1.0))
    log_mu_shots  = pyro.sample("log_mu_shots",  dist.Normal(_LOG_MU_SHOTS,  1.0))
    log_mu_fouled = pyro.sample("log_mu_fouled", dist.Normal(_LOG_MU_FOULED, 1.0))

    # --- Gamma parameters for expectedAssists ---
    xA_alpha = pyro.sample("xA_alpha", dist.HalfNormal(2.0))
    xA_beta  = pyro.sample("xA_beta",  dist.HalfNormal(1.0))

    # --- Dependency: opp_half_passes --> expectedAssists ---
    beta_ohp_xA = pyro.sample("beta_ohp_xA", dist.Normal(0.0, 0.1))

    # --- Dependency: shots_outside --> goals ---
    log_mu_goals       = pyro.sample("log_mu_goals",       dist.Normal(_LOG_MU_GOALS, 1.0))
    beta_shots_goals   = pyro.sample("beta_shots_goals",   dist.Normal(0.0, 0.1))

    # --- Rating regression weights ---
    alpha_rating = pyro.sample("alpha_rating", dist.Normal(0.0, 1.0))
    w_ohp        = pyro.sample("w_ohp",        dist.Normal(0.0, 1.0))
    w_shots      = pyro.sample("w_shots",      dist.Normal(0.0, 1.0))
    w_fouled     = pyro.sample("w_fouled",     dist.Normal(0.0, 1.0))
    w_xA         = pyro.sample("w_xA",         dist.Normal(0.0, 1.0))
    w_goals      = pyro.sample("w_goals",      dist.Normal(0.0, 1.0))
    rating_sigma = pyro.sample("rating_sigma", dist.HalfNormal(1.0))

    # --- Determine batch size ---
    n_obs = 1
    if goals is not None:
        n_obs = goals.shape[0]
    elif rating is not None:
        n_obs = rating.shape[0]

    with pyro.plate("data", n_obs):

        # Root: accurateOppositionHalfPasses ~ NegBin
        ohp_mu  = torch.exp(log_mu_ohp).expand(n_obs)
        ohp_obs = pyro.sample(
            "opp_half_passes",
            dist.NegativeBinomial(r_ohp, r_ohp / (r_ohp + ohp_mu)),
            obs=opp_half_passes,
        )

        # Root: shotsFromOutsideTheBox ~ NegBin
        shots_mu  = torch.exp(log_mu_shots).expand(n_obs)
        shots_obs = pyro.sample(
            "shots_outside",
            dist.NegativeBinomial(r_shots, r_shots / (r_shots + shots_mu)),
            obs=shots_outside,
        )

        # Root: wasFouled ~ NegBin
        fouled_mu  = torch.exp(log_mu_fouled).expand(n_obs)
        fouled_obs = pyro.sample(
            "was_fouled",
            dist.NegativeBinomial(r_fouled, r_fouled / (r_fouled + fouled_mu)),
            obs=was_fouled,
        )

        # Child: expectedAssists ~ Gamma, depends on opp_half_passes
        # Normalise ohp by its prior mean so beta_ohp_xA stays on a unit scale
        xA_concentration = torch.exp(
            torch.log(xA_alpha + 1e-6) + beta_ohp_xA * ohp_obs / _MEAN_OHP
        )
        xA_obs = pyro.sample(
            "xA",
            dist.Gamma(xA_concentration, xA_beta),
            obs=xA,
        )

        # Child: goals ~ NegBin, depends on shots_outside
        goals_log_mu_val = log_mu_goals + beta_shots_goals * shots_obs / _MEAN_SHOTS
        goals_mu_val     = torch.exp(goals_log_mu_val)
        goals_obs        = pyro.sample(
            "goals",
            dist.NegativeBinomial(r_goals, r_goals / (r_goals + goals_mu_val)),
            obs=goals,
        )

        # Rating: Normal with log1p-scaled count inputs
        rating_mu = (
            alpha_rating
            + w_ohp    * torch.log1p(ohp_obs)
            + w_shots  * torch.log1p(shots_obs)
            + w_fouled * torch.log1p(fouled_obs)
            + w_xA     * xA_obs
            + w_goals  * torch.log1p(goals_obs)
        )
        pyro.sample("rating", dist.Normal(rating_mu, rating_sigma), obs=rating)


if __name__ == "__main__":
    df    = load_PL_dataset()
    mf_df = load_midfielder_data(df)

    sub = mf_df[mf_df['minutesPlayed'] >= MIN_MINUTES][FEATURES + ['rating']].dropna()
    print(f"\nMidfielders after >={MIN_MINUTES} min filter: {len(sub)} players")

    def t(col):
        return torch.tensor(sub[col].values, dtype=torch.float32)

    data = {
        'opp_half_passes': t('accurateOppositionHalfPasses'),
        'shots_outside':   t('shotsFromOutsideTheBox'),
        'was_fouled':      t('wasFouled'),
        'xA':              t('expectedAssists'),
        'goals':           t('goals'),
        'rating':          standardize(t('rating')),
    }

    print("\n--- Ancestral Sampling (Prior Predictive Check) ---")
    predictive    = Predictive(midfielder_model, num_samples=1)
    prior_samples = predictive()

    print("Generated 1 sample of fake data from the DAG priors:")
    for k in ['opp_half_passes', 'shots_outside', 'was_fouled', 'xA', 'goals', 'rating']:
        vals = prior_samples[k].flatten()[:5]
        print(f"  {k:<25}: {vals.tolist()}")

    print("\nMidfielder PGM is set up and ready for inference.")
