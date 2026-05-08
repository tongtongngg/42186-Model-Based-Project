"""
Hierarchical Bayesian linear model for EPL player rating prediction.

All four positions are trained jointly. The key difference from the individual
position models is true partial pooling of both intercepts AND weights:

    alpha[g]   ~ Normal(mu_alpha, sigma_alpha)        # hierarchical intercepts
    beta[g, f] ~ Normal(mu_beta[f], sigma_beta[f])    # hierarchical weights

This means positions with fewer players (e.g. GKs) borrow strength from the
global estimate of each feature's importance.

Features are taken from the ARD analysis in hierarchical_ard_feature_selector.py,
with ratio/percentage features excluded.
"""
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from sklearn.preprocessing import StandardScaler

from src.data_utils import load_PL_dataset


POSITIONS = ['D', 'F', 'G', 'M']

FEATURES_PER_POS = {
    'D': ['goalsConcededInsideTheBox', 'goalsConceded', 'duelLost', 'tackles', 'clearances'],
    'F': ['successfulDribbles', 'goalsFromInsideTheBox', 'goals', 'groundDuelsWon', 'expectedAssists'],
    'G': ['goalsPrevented', 'punches', 'goalsConcededInsideTheBox', 'ballRecovery', 'yellowCards'],
    'M': ['expectedAssists', 'accurateLongBalls', 'totalDuelsWon', 'wasFouled', 'duelLost'],
}

ALL_FEATURES = sorted(set(f for feats in FEATURES_PER_POS.values() for f in feats))
NUM_POSITIONS = len(POSITIONS)
NUM_FEATURES  = len(ALL_FEATURES)

# Fixed (G, F) mask: 1 where position g uses feature f.
# This encodes domain knowledge — not learned from data.
_MASK = torch.zeros(NUM_POSITIONS, NUM_FEATURES)
for _g, _pos in enumerate(POSITIONS):
    for _feat in FEATURES_PER_POS[_pos]:
        _MASK[_g, ALL_FEATURES.index(_feat)] = 1.0


def hierarchical_model(X, pos_ids, rating=None):
    """
    Hierarchical Bayesian linear regression over all positions.

    Parameters
    ----------
    X        : (N, F) standardised feature matrix
    pos_ids  : (N,)  integer position indices (0=D, 1=F, 2=G, 3=M)
    rating   : (N,)  standardised ratings, None at prediction time

    alpha and beta are sampled as event tensors (not plates) so that
    Predictive can index into them without shape conflicts.
    """
    G, F = NUM_POSITIONS, NUM_FEATURES

    # --- Hierarchical intercepts: alpha[g] ~ Normal(mu_alpha, sigma_alpha) ---
    mu_alpha    = pyro.sample("mu_alpha",    dist.Normal(0.0, 1.0))
    sigma_alpha = pyro.sample("sigma_alpha", dist.HalfNormal(0.5))
    alpha = pyro.sample(
        "alpha",
        dist.Normal(mu_alpha * torch.ones(G), sigma_alpha * torch.ones(G)).to_event(1),
    )  # event shape (G,)

    # --- Hierarchical weights: beta[g,f] ~ Normal(mu_beta[f], sigma_beta[f]) ---
    # Positions share hyperpriors per feature — this is the partial pooling.
    mu_beta = pyro.sample(
        "mu_beta",
        dist.Normal(torch.zeros(F), 0.5 * torch.ones(F)).to_event(1),
    )  # event shape (F,)
    sigma_beta = pyro.sample(
        "sigma_beta",
        dist.HalfNormal(0.5 * torch.ones(F)).to_event(1),
    )  # event shape (F,)
    beta = pyro.sample(
        "beta",
        dist.Normal(
            mu_beta.unsqueeze(0).expand(G, F),
            sigma_beta.unsqueeze(0).expand(G, F),
        ).to_event(2),
    )  # event shape (G, F)

    sigma_y = pyro.sample("sigma_y", dist.HalfNormal(1.0))

    # Zero out features not selected for each position
    beta_masked = beta * _MASK  # (G, F)
    mu = alpha[pos_ids] + (beta_masked[pos_ids] * X).sum(-1)

    with pyro.plate("data", X.shape[0]):
        pyro.sample("rating", dist.Normal(mu, sigma_y), obs=rating)


def get_hierarchical_data(df, scalers_x=None, scaler_y=None):
    """
    Build model inputs from a DataFrame.

    Returns a kwargs dict ready for ``model(**kwargs)`` and the fitted scalers
    (pass them in on the test set to avoid data leakage).
    """
    data = (
        df[df['position'].isin(POSITIONS)]
        .dropna(subset=['rating'])
        .query("minutesPlayed >= 450")
        .copy()
    )
    data[ALL_FEATURES] = data[ALL_FEATURES].fillna(0)

    pos_ids_np = np.array([POSITIONS.index(p) for p in data['position']], dtype=np.int64)
    X_raw = data[ALL_FEATURES].values.astype(np.float32)
    y_raw = data['rating'].values.astype(np.float32).reshape(-1, 1)

    if scalers_x is None:
        scalers_x = StandardScaler()
        X_proc = scalers_x.fit_transform(X_raw)
    else:
        X_proc = scalers_x.transform(X_raw)

    if scaler_y is None:
        scaler_y = StandardScaler()
        y_proc = scaler_y.fit_transform(y_raw)
    else:
        y_proc = scaler_y.transform(y_raw)

    return (
        {
            'X':       torch.tensor(X_proc,           dtype=torch.float32),
            'pos_ids': torch.tensor(pos_ids_np,       dtype=torch.long),
            'rating':  torch.tensor(y_proc.squeeze(), dtype=torch.float32),
        },
        scalers_x,
        scaler_y,
    )
