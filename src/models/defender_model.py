import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive, MCMC, NUTS
from src.data_utils import load_PL_dataset


def standardize(tensor):
    std = tensor.std()
    if std == 0:
        return tensor - tensor.mean()
    return (tensor - tensor.mean()) / std


def defender_model(
    touches=None,
    accuratePasses=None,
    totalDuelsWon=None,
    clearances=None,
    cleanSheet=None,
    rating=None,
):
    """
    Defender PGM matching the DAG:

        touches ---------> rating
            |
            v
        accuratePasses --> rating

        totalDuelsWon --> clearances --> cleanSheet --> rating

    Notes:
    - continuous variables are standardized
    - cleanSheet is binary 0/1 and modeled with Bernoulli
    """

    # Determine batch size
    n_obs = None
    for x in [touches, accuratePasses, totalDuelsWon, clearances, cleanSheet, rating]:
        if x is not None:
            n_obs = x.shape[0]
            break

    if n_obs is None:
        n_obs = 1

    # Priors for exogenous variables
    touches_mu = pyro.sample("touches_mu", dist.Normal(0, 1))
    touches_sigma = pyro.sample("touches_sigma", dist.HalfNormal(1))

    totalDuelsWon_mu = pyro.sample("totalDuelsWon_mu", dist.Normal(0, 1))
    totalDuelsWon_sigma = pyro.sample("totalDuelsWon_sigma", dist.HalfNormal(1))

    # accuratePasses <- touches
    alpha_passes = pyro.sample("alpha_passes", dist.Normal(0, 1))
    w_touches_to_passes = pyro.sample("w_touches_to_passes", dist.Normal(0, 1))
    passes_sigma = pyro.sample("passes_sigma", dist.HalfNormal(1))

    # clearances <- totalDuelsWon
    alpha_clearances = pyro.sample("alpha_clearances", dist.Normal(0, 1))
    w_duels_to_clearances = pyro.sample("w_duels_to_clearances", dist.Normal(0, 1))
    clearances_sigma = pyro.sample("clearances_sigma", dist.HalfNormal(1))

    # cleanSheet <- clearances
    alpha_cleanSheet = pyro.sample("alpha_cleanSheet", dist.Normal(0, 1))
    w_clearances_to_cleanSheet = pyro.sample("w_clearances_to_cleanSheet", dist.Normal(0, 1))

    # rating <- touches + accuratePasses + cleanSheet
    alpha_rating = pyro.sample("alpha_rating", dist.Normal(0, 1))
    w_touches_to_rating = pyro.sample("w_touches_to_rating", dist.Normal(0, 1))
    w_passes_to_rating = pyro.sample("w_passes_to_rating", dist.Normal(0, 1))
    w_cleanSheet_to_rating = pyro.sample("w_cleanSheet_to_rating", dist.Normal(0, 1))
    rating_sigma = pyro.sample("rating_sigma", dist.HalfNormal(1))

    with pyro.plate("data", n_obs):

        touches_obs = pyro.sample(
            "touches",
            dist.Normal(touches_mu, touches_sigma),
            obs=touches,
        )

        totalDuelsWon_obs = pyro.sample(
            "totalDuelsWon",
            dist.Normal(totalDuelsWon_mu, totalDuelsWon_sigma),
            obs=totalDuelsWon,
        )

        accuratePasses_mu = alpha_passes + w_touches_to_passes * touches_obs

        accuratePasses_obs = pyro.sample(
            "accuratePasses",
            dist.Normal(accuratePasses_mu, passes_sigma),
            obs=accuratePasses,
        )

        clearances_mu = alpha_clearances + w_duels_to_clearances * totalDuelsWon_obs

        clearances_obs = pyro.sample(
            "clearances",
            dist.Normal(clearances_mu, clearances_sigma),
            obs=clearances,
        )

        cleanSheet_logits = (
            alpha_cleanSheet
            + w_clearances_to_cleanSheet * clearances_obs
        )

        cleanSheet_obs = pyro.sample(
            "cleanSheet",
            dist.Bernoulli(logits=cleanSheet_logits),
            obs=cleanSheet,
        )

        rating_mu = (
            alpha_rating
            + w_touches_to_rating * touches_obs
            + w_passes_to_rating * accuratePasses_obs
            + w_cleanSheet_to_rating * cleanSheet_obs
        )

        rating_obs = pyro.sample(
            "rating",
            dist.Normal(rating_mu, rating_sigma),
            obs=rating,
        )

    return rating_obs


if __name__ == "__main__":
    pyro.clear_param_store()

    df = load_PL_dataset()

    def_df = df[df["position"] == "D"].copy()

    data = {
        "touches": torch.tensor(def_df["touches"].values, dtype=torch.float32),
        "accuratePasses": torch.tensor(def_df["accuratePasses"].values, dtype=torch.float32),
        "totalDuelsWon": torch.tensor(def_df["totalDuelsWon"].values, dtype=torch.float32),
        "clearances": torch.tensor(def_df["clearances"].values, dtype=torch.float32),
        "cleanSheet": torch.tensor(def_df["cleanSheet"].values, dtype=torch.float32),
        "rating": torch.tensor(def_df["rating"].values, dtype=torch.float32),
    }

    model_data = {
        "touches": standardize(data["touches"]),
        "accuratePasses": standardize(data["accuratePasses"]),
        "totalDuelsWon": standardize(data["totalDuelsWon"]),
        "clearances": standardize(data["clearances"]),
        "cleanSheet": data["cleanSheet"],  # keep binary, do not standardize
        "rating": standardize(data["rating"]),
    }

    print("--- Prior Predictive Check ---")

    prior_predictive = Predictive(
        defender_model,
        num_samples=100,
    )

    prior_samples = prior_predictive()

    for name in [
        "touches",
        "accuratePasses",
        "totalDuelsWon",
        "clearances",
        "cleanSheet",
        "rating",
    ]:
        print(f"{name}: {prior_samples[name].shape}")
        print(prior_samples[name].flatten()[:5])

    print("\n--- Running NUTS Inference ---")

    nuts_kernel = NUTS(defender_model)

    mcmc = MCMC(
        nuts_kernel,
        num_samples=1000,
        warmup_steps=500,
        num_chains=1,
    )

    mcmc.run(
        touches=model_data["touches"],
        accuratePasses=model_data["accuratePasses"],
        totalDuelsWon=model_data["totalDuelsWon"],
        clearances=model_data["clearances"],
        cleanSheet=model_data["cleanSheet"],
        rating=model_data["rating"],
    )

    print("\n--- Posterior Summary ---")
    mcmc.summary()

    posterior_samples = mcmc.get_samples()

    print("\nPosterior mean effects:")

    for name in [
        "w_touches_to_passes",
        "w_duels_to_clearances",
        "w_clearances_to_cleanSheet",
        "w_touches_to_rating",
        "w_passes_to_rating",
        "w_cleanSheet_to_rating",
    ]:
        print(f"{name}: {posterior_samples[name].mean().item():.3f}")

    print("\nDefender DAG model inference complete.")