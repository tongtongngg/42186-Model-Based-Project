import torch
import pyro
import pyro.distributions as dist
import pandas as pd
import numpy as np
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

from src.data_utils import load_PL_dataset
from src.data_utils.midfield_pre import PLAYTIME_PROXIES


ID_COLS = ['player_name', 'team_name', 'position']


def standardize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.mean()) / tensor.std()


def hierarchical_ard_model(X, pos_ids, num_positions, num_features, y=None):
    """
    Hierarchical Bayesian regression with Automatic Relevance Determination.

    Hierarchy: each position (G/D/M/F) has its own coefficient vector around a
    shared global mean. ARD per-feature precision shrinks irrelevant feature
    weights toward zero.

    Args:
        X: float tensor (N, F) of standardized features
        pos_ids: long tensor (N,) of position indices in [0, num_positions)
        num_positions: int (4 for G/D/M/F)
        num_features: int F
        y: optional float tensor (N,) of standardized ratings
    """

    # ----- ARD: per-feature precision drives irrelevant weights to 0 -----
    with pyro.plate("features_ard", num_features):
        tau = pyro.sample("tau", dist.Gamma(0.5, 0.5))
        # Clamp away from 0 to keep sigma_feat from exploding numerically
        sigma_feat = 1.0 / torch.sqrt(tau.clamp(min=1e-4))
        mu_beta = pyro.sample("mu_beta", dist.Normal(0.0, sigma_feat))

    # ----- Between-position variation around the (ARD-shrunk) global mean -----
    # HalfNormal(1.0) lets positions deviate substantially; tighter priors
    # caused all four positions to collapse onto identical rankings.
    sigma_pos_beta = pyro.sample("sigma_pos_beta", dist.HalfNormal(1.0))

    # ----- Intercept hyperpriors -----
    mu_alpha = pyro.sample("mu_alpha", dist.Normal(0.0, 1.0))
    sigma_alpha = pyro.sample("sigma_alpha", dist.HalfNormal(1.0))

    # ----- Position-level partial pooling -----
    with pyro.plate("positions", num_positions):
        alpha_pos = pyro.sample("alpha_pos", dist.Normal(mu_alpha, sigma_alpha))
        beta_pos = pyro.sample(
            "beta_pos",
            dist.Normal(mu_beta, sigma_pos_beta).to_event(1),
        )

    # ----- Likelihood -----
    sigma_y = pyro.sample("sigma_y", dist.HalfNormal(1.0))
    with pyro.plate("data", X.shape[0]):
        mu = alpha_pos[pos_ids] + (beta_pos[pos_ids] * X).sum(dim=-1)
        pyro.sample("rating", dist.Normal(mu, sigma_y), obs=y)


def preprocess(df: pd.DataFrame):
    """Build standardized X, y, pos_ids from the full PL dataset."""
    df = df.dropna(subset=['rating']).copy()

    pos_cat = pd.Categorical(df['position'])
    pos_ids_np = pos_cat.codes
    pos_categories = list(pos_cat.categories)

    drop_cols = set(ID_COLS) | set(PLAYTIME_PROXIES) | {'rating'}
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c not in drop_cols]

    X_df = numeric_df[feature_cols].fillna(0.0)

    stds = X_df.std(axis=0)
    keep = stds[stds > 0].index.tolist()
    if len(keep) < len(feature_cols):
        dropped = set(feature_cols) - set(keep)
        print(f"Dropping {len(dropped)} zero-variance columns: {sorted(dropped)}")
    X_df = X_df[keep]

    X_std = (X_df - X_df.mean()) / X_df.std()
    y_raw = df['rating'].astype(float).values
    y_std = (y_raw - y_raw.mean()) / y_raw.std()

    X = torch.tensor(X_std.values, dtype=torch.float32)
    y = torch.tensor(y_std, dtype=torch.float32)
    pos_ids = torch.tensor(pos_ids_np, dtype=torch.long)

    return X, y, pos_ids, keep, pos_categories


def _print_top(name, values, names, k=10, reverse=True):
    order = np.argsort(values)
    if reverse:
        order = order[::-1]
    print(f"\n  {name}:")
    for i in order[:k]:
        print(f"    {names[i]:<40} {values[i]:+.4f}")


if __name__ == "__main__":
    pyro.set_rng_seed(0)

    print("--- Loading data ---")
    df = load_PL_dataset()
    X, y, pos_ids, feature_names, pos_categories = preprocess(df)
    N, F = X.shape
    G = len(pos_categories)
    print(f"  N={N} players, F={F} features, G={G} positions ({pos_categories})")

    print("\n--- Prior predictive check ---")
    predictive = Predictive(hierarchical_ard_model, num_samples=1)
    prior_samples = predictive(X, pos_ids, G, F)
    sampled_rating = prior_samples['rating'].flatten()[:5]
    print(f"  First 5 prior-sampled ratings: {sampled_rating.tolist()}")

    print("\n--- Training (SVI + AutoNormal) ---")
    pyro.clear_param_store()
    guide = AutoNormal(hierarchical_ard_model)
    svi = SVI(hierarchical_ard_model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

    num_steps = 3000
    for step in range(num_steps):
        loss = svi.step(X, pos_ids, G, F, y)
        if step % 500 == 0 or step == num_steps - 1:
            print(f"  Step {step:4d} | ELBO loss = {loss:.2f}")

    print("\n--- Posterior summary ---")
    posterior = Predictive(
        hierarchical_ard_model, guide=guide, num_samples=800,
        return_sites=("tau", "mu_beta", "beta_pos", "alpha_pos",
                      "sigma_pos_beta", "sigma_y"),
    )
    samples = posterior(X, pos_ids, G, F)

    tau_mean = samples['tau'].mean(0).detach().numpy()
    sigma_feat_mean = 1.0 / np.sqrt(np.clip(tau_mean, 1e-6, None))
    mu_beta_mean = samples['mu_beta'].mean(0).detach().numpy()
    beta_pos_mean = samples['beta_pos'].mean(0).detach().numpy()
    alpha_pos_mean = samples['alpha_pos'].mean(0).detach().numpy()

    feature_names_arr = np.array(feature_names)

    print("\n[ARD] Top-15 features by sigma_feat = 1/sqrt(tau) (kept by ARD):")
    _print_top("largest sigma_feat", sigma_feat_mean, feature_names_arr, k=15, reverse=True)

    print("\n[ARD] Bottom-10 features by sigma_feat (shrunk out by ARD):")
    _print_top("smallest sigma_feat", sigma_feat_mean, feature_names_arr, k=10, reverse=False)

    print("\n[Global] Top-10 positive global weights mu_beta:")
    _print_top("mu_beta+", mu_beta_mean, feature_names_arr, k=10, reverse=True)

    print("\n[Global] Top-10 negative global weights mu_beta:")
    _print_top("mu_beta-", mu_beta_mean, feature_names_arr, k=10, reverse=False)

    print("\n[Hierarchy] Per-position top-5 features by |beta_pos|:")
    for g, pos_label in enumerate(pos_categories):
        abs_w = np.abs(beta_pos_mean[g])
        order = np.argsort(abs_w)[::-1][:5]
        print(f"  Position {pos_label} (alpha={alpha_pos_mean[g]:+.3f}):")
        for i in order:
            print(f"    {feature_names_arr[i]:<40} {beta_pos_mean[g, i]:+.4f}")

    print("\nDone.")
