import os
import argparse
import torch
import pyro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from src.data_utils import load_PL_dataset

# Import models
from src.models.attacker_model import model as attacker_model_fn

def preprocess_with_mapping(df, features, target, team_mapping=None, scaler_x=None, scaler_y=None):
    """
    Standardizes features and encodes team names consistently.
    """
    data = df.copy()
    data[features] = data[features].fillna(0)
    
    if scaler_x is None:
        scaler_x = StandardScaler()
        X_scaled = scaler_x.fit_transform(data[features])
    else:
        X_scaled = scaler_x.transform(data[features])
        
    if scaler_y is None:
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(data[[target]])
    else:
        y_scaled = scaler_y.transform(data[[target]])

    X = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(y_scaled, dtype=torch.float).squeeze()

    if team_mapping is None:
        team_names = sorted(data['team_name'].unique())
        team_mapping = {name: i for i, name in enumerate(team_names)}
    
    team_ids = torch.tensor(data['team_name'].map(team_mapping).values, dtype=torch.long)
    num_teams = len(team_mapping)

    return X, y, team_ids, num_teams, team_mapping, scaler_x, scaler_y

def plot_predictions(y_true, y_pred, y_std, model_name):
    # plotting section
    plt.figure(figsize=(10, 7))
    plt.errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.5, label='Preds (Mean ± SD)', capsize=3)
    
    min_val = min(y_true.min(), y_pred.min()) - 0.5
    max_val = max(y_true.max(), y_pred.max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel("Actual Standardized Rating")
    plt.ylabel("Predicted Standardized Rating")
    plt.title(f"{model_name.capitalize()} Model: Predictions vs Actual (Test Set)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    output_plot = os.path.join(plot_dir, f"{model_name}_eval_plot.png")
    plt.savefig(output_plot)
    print(f"Evaluation plot saved to {output_plot}")

def run_evaluation(model_name, num_steps=2000, lr=0.01):
    """
    Evaluates the specified model and saves performance plots.
    """
    print(f"--- Evaluating {model_name.capitalize()} Model ---")
    
    # Configuration based on model
    if model_name == "attacker":
        model_fn = attacker_model_fn
        features = ['totalAttemptAssist', 'groundDuelsWon', 'keyPasses', 'goals']
        pos_filter = 'F'
    else:
        raise ValueError(f"Model '{model_name}' evaluation not yet implemented.")

    target = 'rating'
    

    df = load_PL_dataset()
    pos_df = df[df['position'] == pos_filter].copy()
    train_df, test_df = train_test_split(pos_df, test_size=0.2, random_state=42)
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    X_train, y_train, team_ids_train, num_teams, team_mapping, scaler_x, scaler_y = preprocess_with_mapping(
        train_df, features, target
    )
    X_test, y_test, team_ids_test, _, _, _, _ = preprocess_with_mapping(
        test_df, features, target, team_mapping=team_mapping, scaler_x=scaler_x, scaler_y=scaler_y
    )
    num_features = len(features)

    pyro.clear_param_store()
    guide = AutoNormal(model_fn)
    optimizer = Adam({"lr": lr})
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    print(f"Training on {len(train_df)} samples...")
    for step in range(num_steps):
        loss = svi.step(team_ids_train, X_train, num_teams, num_features, y_train)
        if step % 500 == 0:
            print(f"Step {step:4d} : Loss = {loss:.4f}")

    print("\nGenerating predictions on test set...")
    predictive = Predictive(model_fn, guide=guide, num_samples=500)
    test_samples = predictive(team_ids_test, X_test, num_teams, num_features)
    
    y_pred = test_samples['obs'].mean(axis=0).detach().numpy()
    y_std = test_samples['obs'].std(axis=0).detach().numpy()
    y_true = y_test.detach().numpy()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n" + "="*30)
    print(f"      {model_name.upper()} PERFORMANCE")
    print("="*30)
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 : {r2:.4f}")
    print("="*30)

    plot_predictions(y_true, y_pred, y_std, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate performance of MBML models.")
    parser.add_argument("model", choices=["attacker"], help="Name of the model to evaluate.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of SVI steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    
    args = parser.parse_args()
    
    run_evaluation(args.model, num_steps=args.steps, lr=args.lr)
