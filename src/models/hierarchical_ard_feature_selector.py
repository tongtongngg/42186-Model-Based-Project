import torch
import pyro
import pyro.distributions as dist
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam

from src.data_utils import load_PL_dataset
from src.data_utils.midfield_pre import PLAYTIME_PROXIES


ID_COLS = ['player_name', 'team_name', 'position']

# Percentage/ratio features — already scale-normalized so they dominate the
# global ARD despite being no more informative than their raw counterparts.
RATIO_FEATURES = {
    'totalDuelsWonPercentage',
    'aerialDuelsWonPercentage',
    'groundDuelsWonPercentage',
    'accuratePassesPercentage',
    'accurateLongBallsPercentage',
    'accurateCrossesPercentage',
    'successfulDribblesPercentage',
    'goalConversionPercentage',
    'tacklesWonPercentage'
}


def standardize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.mean()) / tensor.std()


def select_positional_features(
    df: pd.DataFrame,
    pos_categories: list,
    feature_cols: list,
    threshold: float = 0.2,
) -> list:
    """Keep features with |Spearman r| >= threshold with rating for at least one position group."""
    selected = []
    for feat in feature_cols:
        for pos in pos_categories:
            pos_df = df[df['position'] == pos].dropna(subset=['rating'])
            if len(pos_df) < 5:
                continue
            x_pos = pos_df[feat].fillna(0).values
            if np.std(x_pos) < 1e-6:
                continue
            r, _ = spearmanr(x_pos, pos_df['rating'].values)
            if abs(r) >= threshold:
                selected.append(feat)
                break
    return selected


def positional_relevance(
    df: pd.DataFrame,
    pos: str,
    feature_cols: list,
    threshold: float = 0.15,
) -> list:
    """Return indices of features with |Spearman r| >= threshold for a single position."""
    pos_df = df[df['position'] == pos].dropna(subset=['rating'])
    relevant = []
    for f, feat in enumerate(feature_cols):
        x_pos = pos_df[feat].fillna(0).values
        if np.std(x_pos) < 1e-6 or len(pos_df) < 5:
            continue
        r, _ = spearmanr(x_pos, pos_df['rating'].values)
        if abs(r) >= threshold:
            relevant.append(f)
    return relevant


def hierarchical_ard_model(X, pos_ids, num_positions, num_features, y=None):
    # 1. Global Intercept
    mu_alpha = pyro.sample("mu_alpha", dist.Normal(0.0, 1.0))
    sigma_alpha = pyro.sample("sigma_alpha", dist.HalfNormal(1.0))

    with pyro.plate("positions_plate", num_positions):
        alpha_pos = pyro.sample("alpha_pos", dist.Normal(mu_alpha, sigma_alpha))

    with pyro.plate("features", num_features):
        global_tau = pyro.sample("global_tau", dist.Gamma(2.0, 1.0))
        global_scale = 1.0 / torch.sqrt(global_tau.clamp(min=1e-4))

        with pyro.plate("pos_ard", num_positions):
            local_tau = pyro.sample("local_tau", dist.Gamma(2.0, 1.0))
            local_scale = 1.0 / torch.sqrt(local_tau.clamp(min=1e-4))
            epsilon = pyro.sample("epsilon", dist.Normal(0.0, 1.0))

    beta_pos = (global_scale.unsqueeze(0) * local_scale) * epsilon

    # 4. Likelihood
    sigma_y = pyro.sample("sigma_y", dist.HalfNormal(1.0))

    # beta_pos 
    current_beta = beta_pos[pos_ids, :]
    
    mu = alpha_pos[pos_ids] + (current_beta * X).sum(dim=-1)
    
    with pyro.plate("data", X.shape[0]):
        pyro.sample("rating", dist.Normal(mu, sigma_y), obs=y)


def preprocess(df: pd.DataFrame, pos_threshold: float = 0.2):
    """Build standardized X, y, pos_ids from the full PL dataset.

    Only features with |Spearman r| >= pos_threshold for at least one position
    are kept, so GK-only stats don't pollute the outfield feature space.
    """
    df = df.dropna(subset=['rating']).copy()

    pos_cat = pd.Categorical(df['position'])
    pos_ids_np = pos_cat.codes
    pos_categories = list(pos_cat.categories)

    drop_cols = set(ID_COLS) | set(PLAYTIME_PROXIES) | RATIO_FEATURES | {'rating'}
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c not in drop_cols]

    X_df = numeric_df[feature_cols].fillna(0.0)

    stds = X_df.std(axis=0)
    nonzero = stds[stds > 0].index.tolist()
    if len(nonzero) < len(feature_cols):
        print(f"Dropping {len(feature_cols) - len(nonzero)} zero-variance columns")

    keep = select_positional_features(df, pos_categories, nonzero, threshold=pos_threshold)
    print(f"  {len(keep)}/{len(nonzero)} features pass |r| >= {pos_threshold} for at least one position")
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
    NUM_STEPS = 20000
    pyro.clear_param_store()
    guide = AutoNormal(hierarchical_ard_model)
    svi = SVI(
        hierarchical_ard_model, guide,
        ClippedAdam({"lr": 0.01, "lrd": 0.1 ** (1 / NUM_STEPS), "clip_norm": 10.0}),
        loss=Trace_ELBO(),
    )

    print(f"Training Hierarchical ARD on {N} players...")
    for step in range(NUM_STEPS):
        loss = svi.step(X, pos_ids, G, F, y)
        if step % 5000 == 0:
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

    # Global relevance — only features relevant to multiple positions make sense here
    print("\n[GLOBAL ARD] Top 10 metrics across all positions:")
    _print_top("Metric Importance", global_scale, feature_names_arr, k=10)

    # Position specific drivers — only show features that correlate with rating
    # for that position (suppresses GK stats showing up for outfield roles)
    print("\n[POSITIONAL ANALYSIS] Key predictive drivers per role:")
    for g, pos_label in enumerate(pos_categories):
        pos_weights = beta_pos_mean[g, :]
        relevant_idx = positional_relevance(df, pos_label, feature_names, threshold=0.15)
        print(f"\n>> {pos_label} (Baseline: {alpha_pos_mean[g]:+.2f})")
        if not relevant_idx:
            print("   (no features passed relevance threshold)")
            continue
        order = sorted(relevant_idx, key=lambda i: abs(pos_weights[i]), reverse=True)[:5]
        for i in order:
            print(f"   {pos_weights[i]:+.4f} | {feature_names_arr[i]}")

    print("\n" + "="*50)
    print("Process Complete.")
