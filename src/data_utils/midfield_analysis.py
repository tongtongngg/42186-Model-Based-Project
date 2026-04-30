import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.data_utils.load_dataset import load_PL_dataset
from src.data_utils.midfield_pre import load_midfielder_data

FEATURES = [
    'goals',
    'aerialDuelsWon',
    'accurateOppositionHalfPasses',
    'wasFouled',
    'expectedAssists',
    'successfulDribbles',
    'shotsFromOutsideTheBox',
]
COUNT_FEATURES = [f for f in FEATURES if f != 'expectedAssists']


def print_summary_stats(df: pd.DataFrame, features: list) -> None:
    print("\n--- Summary Statistics ---")
    header = f"  {'Feature':<40} {'mean':>7} {'var':>9} {'std':>7} {'skew':>7} {'kurt':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for f in features:
        col = df[f].dropna()
        print(
            f"  {f:<40} "
            f"{col.mean():>7.3f} "
            f"{col.var():>9.3f} "
            f"{col.std():>7.3f} "
            f"{stats.skew(col):>7.3f} "
            f"{stats.kurtosis(col):>7.3f}"
        )


def run_normality_tests(df: pd.DataFrame, features: list) -> None:
    print("\n--- Normality Tests (Shapiro-Wilk & Anderson-Darling, alpha=0.05) ---")
    print(f"  {'Feature':<40} {'SW p-val':>10} {'Normal?':>8}  {'AD stat':>8} {'AD crit(5%)':>12} {'Normal?':>8}")
    print("  " + "-" * 95)
    for f in features:
        col = df[f].dropna().values
        sw_stat, sw_p = stats.shapiro(col)
        ad_result = stats.anderson(col, dist='norm')
        ad_stat = ad_result.statistic
        # critical value index 2 = 5% significance level
        ad_crit = ad_result.critical_values[2]
        sw_ok = "YES" if sw_p > 0.05 else "NO"
        ad_ok = "YES" if ad_stat < ad_crit else "NO"
        print(
            f"  {f:<40} {sw_p:>10.4f} {sw_ok:>8}  {ad_stat:>8.3f} {ad_crit:>12.3f} {ad_ok:>8}"
        )


def run_poisson_tests(df: pd.DataFrame, features: list) -> None:
    print("\n--- Poisson Goodness-of-Fit (count features only) ---")
    print(f"  {'Feature':<40} {'lambda':>8} {'mean/var':>9} {'chi2 p':>8}  {'Verdict'}")
    print("  " + "-" * 85)
    for f in features:
        col = df[f].dropna().values.astype(int)
        lam = col.mean()
        disp = col.var() / col.mean() if col.mean() > 0 else float('nan')

        # Chi-squared test: bin observed vs Poisson-expected counts
        max_val = int(col.max())
        observed, bin_edges = np.histogram(col, bins=range(0, max_val + 2))
        bins = np.arange(0, max_val + 1)
        expected = stats.poisson.pmf(bins, lam) * len(col)

        # Merge tail bins with expected < 5 to satisfy chi2 assumptions
        while len(expected) > 1 and expected[-1] < 5:
            observed[-2] += observed[-1]
            expected[-2] += expected[-1]
            observed = observed[:-1]
            expected = expected[:-1]

        chi2_stat, chi2_p = stats.chisquare(observed, f_exp=expected)

        if chi2_p > 0.05 and abs(disp - 1.0) < 0.3:
            verdict = "Poisson fit"
        elif disp > 1.3:
            verdict = "Overdispersed -> Negative Binomial"
        else:
            verdict = "Poor Poisson fit"

        print(f"  {f:<40} {lam:>8.3f} {disp:>9.3f} {chi2_p:>8.4f}  {verdict}")


def plot_histograms_with_fits(df: pd.DataFrame, features: list) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes_flat = axes.flatten()

    for i, f in enumerate(features):
        ax = axes_flat[i]
        col = df[f].dropna()

        if f in COUNT_FEATURES:
            sns.histplot(col, ax=ax, stat='density', discrete=True,
                         color='steelblue', alpha=0.6, label='Observed')
            lam = col.mean()
            x_vals = np.arange(0, int(col.max()) + 1)
            pmf = stats.poisson.pmf(x_vals, lam)
            ax.vlines(x_vals, 0, pmf, colors='tomato', linewidth=1.8, label=f'Poisson(λ={lam:.2f})')
            ax.set_title(f)
            ax.legend(fontsize=8)
        else:
            sns.histplot(col, ax=ax, stat='density', kde=True,
                         color='steelblue', alpha=0.6, label='Observed + KDE')
            x_plot = np.linspace(col.min(), col.max(), 300)
            # Normal fit
            mu, sigma = stats.norm.fit(col)
            ax.plot(x_plot, stats.norm.pdf(x_plot, mu, sigma),
                    'tomato', lw=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
            # Gamma fit
            a, loc, scale = stats.gamma.fit(col, floc=0)
            ax.plot(x_plot, stats.gamma.pdf(x_plot, a, loc, scale),
                    'green', lw=2, linestyle='--', label=f'Gamma(α={a:.2f})')
            ax.set_title(f)
            ax.legend(fontsize=8)

        ax.set_xlabel(f)
        ax.set_ylabel('Density')

    axes_flat[-1].set_visible(False)
    fig.suptitle('Midfielder Feature Distributions with Fitted Models', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('data/midfield_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved -> data/midfield_distributions.png")


def plot_qq_plots(df: pd.DataFrame, features: list) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes_flat = axes.flatten()

    for i, f in enumerate(features):
        ax = axes_flat[i]
        col = df[f].dropna().values
        stats.probplot(col, dist='norm', plot=ax)
        ax.set_title(f'Q-Q: {f}')
        ax.get_lines()[0].set(markersize=3, alpha=0.6)

    axes_flat[-1].set_visible(False)
    fig.suptitle('Normal Q-Q Plots — Midfielder Features', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('data/midfield_qq.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved -> data/midfield_qq.png")


def plot_violin_plots(df: pd.DataFrame, features: list) -> None:
    # Z-score each feature so all fit on one axis
    z_df = pd.DataFrame({f: stats.zscore(df[f].dropna()) for f in features})
    melted = z_df.melt(var_name='Feature', value_name='z-score')

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(data=melted, x='Feature', y='z-score', hue='Feature',
                   ax=ax, inner='box', palette='Set2', cut=0, legend=False)
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax.tick_params(axis='x', labelrotation=25, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.set_title('Midfielder Feature Distributions (z-scored) — Violin Plots')
    ax.set_ylabel('z-score')
    plt.tight_layout()
    plt.savefig('data/midfield_violins.png', dpi=150)
    plt.show()
    print("  Saved -> data/midfield_violins.png")


if __name__ == "__main__":
    df = load_PL_dataset()
    mf_df = load_midfielder_data(df)
    sub = mf_df[FEATURES].dropna()
    print(f"\nAnalysing {len(sub)} midfielders with complete data across all 7 features.")

    print_summary_stats(sub, FEATURES)
    run_normality_tests(sub, FEATURES)
    run_poisson_tests(sub, COUNT_FEATURES)
    plot_histograms_with_fits(sub, FEATURES)
    plot_qq_plots(sub, FEATURES)
    plot_violin_plots(sub, FEATURES)
