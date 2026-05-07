import torch
import pyro
import pyro.distributions as dist
import pandas as pd
import numpy as np
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

from src.data_utils import load_PL_dataset
from src.data_utils.midfield_pre import PLAYTIME_PROXIES


ID_COLS = ['player_name', 'team_name', 'position']


def standardize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.mean()) / tensor.std()


def hierarchical_ard_model(X, pos_ids, num_positions, num_features, y=None):
    # 1. Global Intercept
    mu_alpha = pyro.sample("mu_alpha", dist.Normal(0.0, 1.0))
    sigma_alpha = pyro.sample("sigma_alpha", dist.HalfNormal(1.0))

    with pyro.plate("positions_plate", num_positions):
        alpha_pos = pyro.sample("alpha_pos", dist.Normal(mu_alpha, sigma_alpha))

    with pyro.plate("features", num_features):
        global_tau = pyro.sample("global_tau", dist.Gamma(0.1, 0.1))
        global_scale = 1.0 / torch.sqrt(global_tau.clamp(min=1e-4))

        with pyro.plate("pos_ard", num_positions):
            local_tau = pyro.sample("local_tau", dist.Gamma(0.1, 0.1))
            local_scale = 1.0 / torch.sqrt(local_tau.clamp(min=1e-4))
            epsilon = pyro.sample("epsilon", dist.Normal(0.0, 1.0))

    # beta_pos is (G, F): local_scale/epsilon are (G, F), global_scale is (F,) → unsqueeze(0) → (1, F).
    beta_pos = (global_scale.unsqueeze(0) * local_scale) * epsilon

    # 4. Likelihood
    sigma_y = pyro.sample("sigma_y", dist.HalfNormal(1.0))

    # beta_pos is (G, F). Indexing [pos_ids, :] gives (N, F) directly.
    current_beta = beta_pos[pos_ids, :]
    
    mu = alpha_pos[pos_ids] + (current_beta * X).sum(dim=-1)
    
    with pyro.plate("data", X.shape[0]):
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

    print("--- Loading & Preprocessing ---")
    df = load_PL_dataset()
    X, y, pos_ids, feature_names, pos_categories = preprocess(df)
    N, F = X.shape
    G = len(pos_categories)
    feature_names_arr = np.array(feature_names)

    # 2. Quiet Training
    pyro.clear_param_store()
    guide = AutoNormal(hierarchical_ard_model)
    svi = SVI(hierarchical_ard_model, guide, Adam({"lr": 0.001}), loss=Trace_ELBO())

    print(f"Training Hierarchical ARD on {N} players...")
    for step in range(10000):
        loss = svi.step(X, pos_ids, G, F, y)
        if step % 2500 == 0:
            print(f"  Step {step:5d} | ELBO Loss: {loss:.2f}")

    # 3. Posterior Summary — sample directly from guide to avoid Predictive's nested-plate issues
    S = 800
    with torch.no_grad():
        guide_draws = [guide(X, pos_ids, G, F) for _ in range(S)]

    global_tau_s = torch.stack([d['global_tau'] for d in guide_draws])  # (S, F)
    local_tau_s  = torch.stack([d['local_tau']  for d in guide_draws])  # (S, G, F)
    epsilon_s    = torch.stack([d['epsilon']     for d in guide_draws])  # (S, G, F)
    alpha_pos_s  = torch.stack([d['alpha_pos']  for d in guide_draws])  # (S, G)

    global_scale_s = 1.0 / torch.sqrt(global_tau_s.clamp(min=1e-4))         # (S, F)
    local_scale_s  = 1.0 / torch.sqrt(local_tau_s.clamp(min=1e-4))          # (S, G, F)
    beta_pos_s = global_scale_s.unsqueeze(1) * local_scale_s * epsilon_s    # (S, G, F)

    global_scale   = global_scale_s.mean(0).numpy()   # (F,)
    beta_pos_mean  = beta_pos_s.mean(0).numpy()        # (G, F)
    alpha_pos_mean = alpha_pos_s.mean(0).numpy()       # (G,)

    print("\n" + "="*50)
    print("STATISTICAL INSIGHTS SUMMARY")
    print("="*50)

    # Global relevance tells you which stats are important across the board
    print("\n[GLOBAL ARD] Top 10 metrics across all positions:")
    _print_top("Metric Importance", global_scale, feature_names_arr, k=10)

    # Position specific drivers
    print("\n[POSITIONAL ANALYSIS] Key predictive drivers per role:")
    for g, pos_label in enumerate(pos_categories):
        # beta_pos_mean is (G, F); index row for this position
        pos_weights = beta_pos_mean[g, :]
        print(f"\n>> {pos_label} (Baseline: {alpha_pos_mean[g]:+.2f})")
        
        # Sort by absolute impact
        order = np.argsort(np.abs(pos_weights))[::-1][:5]
        for i in order:
            print(f"   {pos_weights[i]:+.4f} | {feature_names_arr[i]}")

    print("\n" + "="*50)
    print("Process Complete.")
