from src.data_utils import load_PL_dataset
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from sklearn.preprocessing import StandardScaler

def HierarchicalTeamsModel(team_ids, X, num_teams, num_features, y=None):
    """
    This model was an experimental attempt to incorporate team-level effects into the attacker model using hierarchical modeling.
    """
    # Global Hyper-priors
    mu_beta = pyro.sample("mu_beta", dist.Normal(torch.zeros(num_features), 1.0).to_event(1))
    sigma_beta = pyro.sample("sigma_beta", dist.HalfNormal(torch.ones(num_features)).to_event(1))
    
    # Global Intercept
    mu_alpha = pyro.sample("mu_alpha", dist.Normal(0., 1.))
    sigma_alpha = pyro.sample("sigma_alpha", dist.HalfNormal(1.))

    # Team-level Plate (Partial Pooling)
    with pyro.plate("teams", num_teams):
        team_betas = pyro.sample("team_betas", dist.Normal(mu_beta, sigma_beta).to_event(1))
        team_alphas = pyro.sample("team_alphas", dist.Normal(mu_alpha, sigma_alpha))

    # Individual Observation Plate
    with pyro.plate("data", X.shape[0]):
        linear_combination = (team_betas[team_ids] * X).sum(dim=-1) + team_alphas[team_ids]
        
        pyro.sample("obs", dist.Normal(linear_combination, 0.1), obs=y)


def preprocess_data(df, features, target):
    """
    Filters for attackers, handles missing values, scales features, 
    and prepares PyTorch tensors.
    """
    attackers = df[df['position'] == 'F'].copy()
    attackers[features] = attackers[features].fillna(0)
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X = torch.tensor(scaler_x.fit_transform(attackers[features]), dtype=torch.float)
    y = torch.tensor(scaler_y.fit_transform(attackers[[target]]), dtype=torch.float).squeeze()

    attackers['team_id'] = attackers['team_name'].astype('category').cat.codes
    team_ids = torch.tensor(attackers['team_id'].values, dtype=torch.long)
    
    num_teams = len(attackers['team_name'].unique())
    team_names = attackers['team_name'].astype('category').cat.categories

    return X, y, team_ids, num_teams, team_names


def train_model(model_fn, guide_fn, team_ids, X, num_teams, num_features, y, num_steps=2000, lr=0.01):
    """
    Sets up the optimizer and SVI, clears the parameter store, 
    and runs the training loop.
    """
    pyro.clear_param_store()

    optimizer = Adam({"lr": lr}) 
    
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())

    print(f"Starting inference for {num_steps} steps...")
    for step in range(num_steps):
        loss = svi.step(team_ids, X, num_teams, num_features, y)
        if step % 500 == 0:
            print(f"Step {step:4d} : Loss = {loss:.4f}")
            
    print("Inference complete.")
    return svi


if __name__ == "__main__":
    features = ['totalAttemptAssist', 'groundDuelsWon', 'keyPasses', 'goals']
    target = 'rating'
    num_features = len(features)
    
    df = load_PL_dataset()
    X, y, team_ids, num_teams, team_names = preprocess_data(df, features, target)

    guide = AutoNormal(HierarchicalTeamsModel) 
    train_model(HierarchicalTeamsModel, guide, team_ids, X, num_teams, num_features, y, num_steps=2000)

    predictive = Predictive(HierarchicalTeamsModel, guide=guide, num_samples=800)
    samples = predictive(team_ids, X, num_teams, num_features)
    team_weights = samples['team_betas'].mean(axis=0)

    print(f"\nTop 3 Teams where '{features[0]}' matters MOST for Rating:")
    
    weights_df = pd.DataFrame(team_weights.detach().numpy(), index=team_names, columns=features)
    print(weights_df[features[0]].sort_values(ascending=False).head(3))