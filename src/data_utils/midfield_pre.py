import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
from src.data_utils.load_dataset import load_PL_dataset
from src.data_utils.correlation import get_correlations_by_position

# Columns that only exist for goalkeepers — always NaN for outfield players
GK_ONLY_COLS = [

    'saves', 'savesCaught', 'savesParried',
    'savedShotsFromInsideTheBox', 'savedShotsFromOutsideTheBox',
    'punches', 'highClaims', 'penaltySave', 'penaltyFaced', 'goalsPrevented',
]

# Playing-time proxies: correlate with rating because better players play more,
# not because they cause better ratings — exclude from model features
PLAYTIME_PROXIES = {
      # Playing time
      'minutesPlayed', 'matchesStarted', 'appearances',
      # Leaky / circular
      'totwAppearances',
      # Team-level confounds
      'cleanSheet', 'goalsConcededOutsideTheBox',
      # Volume accumulation (scale with minutes, not skill)
      'rightFootGoals', 'assists',
      'inaccuratePasses', 'possessionLost', 'touches', 'goalsAssistsSum',
  }


def load_midfielder_data(df: pd.DataFrame) -> pd.DataFrame:
    mf = df[df['position'] == 'M'].copy()
    cols_to_drop = [c for c in GK_ONLY_COLS if c in mf.columns]
    mf = mf.drop(columns=cols_to_drop)
    return mf


def analyze_missing_values(df: pd.DataFrame) -> None:
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    print("\n--- Missing Values ---")
    if null_counts.empty:
        print("No missing values.")
    else:
        for col, count in null_counts.items():
            print(f"  {col:<45} {count} nulls")


def get_midfielder_correlations(df: pd.DataFrame, cutoff: float = 0.5) -> dict:
    return get_correlations_by_position(df, positions=['M'], cutoff=cutoff)['M']


def plot_top_correlations(corr_dict: dict, n: int = 15) -> None:
    corr_series = pd.Series(corr_dict).sort_values(key=abs, ascending=False).head(n)
    colors = ['steelblue' if v > 0 else 'tomato' for v in corr_series.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(corr_series.index[::-1], corr_series.values[::-1], color=colors[::-1])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Spearman Correlation with Rating")
    ax.set_title(f"Top {n} Midfielder Stats Correlated with Rating")
    plt.tight_layout()
    plt.savefig('data/midfielder_correlations.png', dpi=150)
    plt.show()


def plot_inter_variable_correlation(df: pd.DataFrame, features: list) -> None:
    sub = df[features].dropna()
    corr_matrix = sub.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap='RdBu_r',
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        annot=False,
    )
    ax.set_title("Inter-Variable Spearman Correlation (Midfielder Candidates)")
    plt.tight_layout()
    plt.savefig('data/midfielder_inter_correlation.png', dpi=150)
    plt.show()
    print("  Saved → data/midfielder_inter_correlation.png")


def compute_mutual_information(df: pd.DataFrame, features: list) -> pd.Series:
    sub = df[features + ['rating']].dropna()
    X = sub[features].values
    y = sub['rating'].values

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=features).sort_values(ascending=False)

    print("\n--- Mutual Information with Rating ---")
    for feat, score in mi_series.items():
        print(f"  {feat:<45} {score:.4f}")

    return mi_series


def select_features_by_clustering(corr_dict: dict, df: pd.DataFrame, n_clusters: int = 7) -> list:
    features = list(corr_dict.keys())
    sub = df[features].dropna()
    abs_corr = sub.corr(method='spearman').abs()

    # Distance matrix: variables that move together have distance near 0
    dist_arr = (1 - abs_corr).values.copy()
    np.fill_diagonal(dist_arr, 0)
    condensed = squareform(dist_arr, checks=False)

    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    cluster_map: dict[int, list] = {}
    for feat, cluster_id in zip(features, labels):
        cluster_map.setdefault(int(cluster_id), []).append(feat)

    print(f"\n--- Hierarchical Clustering → {n_clusters} Clusters ---")
    selected = []
    for cluster_id in sorted(cluster_map):
        cluster_feats = cluster_map[cluster_id]
        best = max(cluster_feats, key=lambda f: abs(corr_dict[f]))
        selected.append(best)
        others = [f for f in cluster_feats if f != best]
        dropped_str = f", dropped {others}" if others else ""
        print(f"  Cluster {cluster_id}: kept '{best}' (r={corr_dict[best]:+.4f}){dropped_str}")

    print(f"\n--- Recommended Features for Midfielder Model ---")
    for f in selected:
        print(f"  {f:<45} r={corr_dict[f]:+.4f}")

    return selected


if __name__ == "__main__":
    df = load_PL_dataset()

    mf_df = load_midfielder_data(df)
    print(f"\nMidfielder data: {mf_df.shape[0]} rows × {mf_df.shape[1]} columns")

    analyze_missing_values(mf_df)

    rating = mf_df['rating'].dropna()
    print("\n--- Rating Distribution ---")
    print(f"  mean={rating.mean():.2f}  std={rating.std():.2f}  "
          f"min={rating.min():.2f}  max={rating.max():.2f}")

    print("\n--- Spearman Correlations with Rating (|r| >= 0.5) ---")
    corr = get_midfielder_correlations(mf_df, cutoff=0.5)
    for var, r in corr.items():
        print(f"  {var:<45} {r:+.4f}")

    plot_top_correlations(corr, n=15)

    candidates = {v: r for v, r in corr.items() if v not in PLAYTIME_PROXIES}

    print("\n--- Inter-Variable Correlation Heatmap ---")
    plot_inter_variable_correlation(mf_df, list(candidates.keys()))

    compute_mutual_information(mf_df, list(candidates.keys()))

    select_features_by_clustering(candidates, mf_df, n_clusters=7)

    """"
    --- Recommended Features for Midfielder Model ---
    goals                                         r=+0.6454
    aerialDuelsWon                                r=+0.5406
    accurateOppositionHalfPasses                  r=+0.7592
    wasFouled                                     r=+0.6632
    expectedAssists                               r=+0.7391
    successfulDribbles                            r=+0.5812
    shotsFromOutsideTheBox                        r=+0.6548
    """
