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
from src.models.attacker_model_basic import forward_model as attacker_model_fn
from src.models.goalkeeper_model import goalkeeper_model as goalkeeper_model_fn
from src.models.midfield_model import midfielder_model as midfielder_model_fn

# Configuration for different models
MODEL_CONFIGS = {
    "attacker": {
        "model_fn": attacker_model_fn,
        "features": ['groundDuelsWon', 'ballRecovery', 'keyPasses', 'expectedAssists', 'totalShots', 'shotsOnTarget', 'goals'],
        "pos_filter": 'F',
        "feature_map": {
            'groundDuelsWon': 'dw',
            'ballRecovery': 'br',
            'keyPasses': 'kp',
            'expectedAssists': 'xa',
            'totalShots': 'ts',
            'shotsOnTarget': 'sot',
            'goals': 'g_raw'
        },
        "target": "rating",
        # attacker_model_basic standardizes these manually or expects them standardized
        "to_standardize": ['groundDuelsWon', 'ballRecovery', 'keyPasses', 'expectedAssists', 'totalShots', 'shotsOnTarget']
    },
    "goalkeeper": {
        "model_fn": goalkeeper_model_fn,
        "features": ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented', 'cleanSheet'],
        "pos_filter": 'G',
        "feature_map": {
            'saves': 'saves',
            'accuratePasses': 'accuratePasses',
            'ballRecovery': 'ballRecovery',
            'goalsPrevented': 'goalsPrevented',
            'cleanSheet': 'cleanSheet_raw'
        },
        "target": "rating",
        "to_standardize": ['saves', 'accuratePasses', 'ballRecovery', 'goalsPrevented']
    },
    "midfielder": {
        "model_fn": midfielder_model_fn,
        "features": ["accurateOppositionHalfPasses", "shotsFromOutsideTheBox", "wasFouled", "expectedAssists", "goals"],
        "pos_filter": 'M',
        "feature_map": {
            'accurateOppositionHalfPasses': 'opp_half_passes',
            'shotsFromOutsideTheBox': 'shots_outside',
            'wasFouled': 'was_fouled',
            'expectedAssists': 'xA',
            'goals': 'goals'
        },
        "target": "rating",
        "to_standardize": [] 
    }
}

def get_model_data(df, config, scalers_x=None, scaler_y=None):
    """
    Prepares data dictionary for Pyro models based on config.
    """
    features = config["features"]
    target = config["target"]
    feature_map = config["feature_map"]
    to_standardize = config.get("to_standardize", [])
    
    data = df.copy()
    data[features] = data[features].fillna(0)
    
    X_dict = {}
    new_scalers_x = scalers_x if scalers_x else {}
    
    for col in features:
        arg_name = feature_map[col]
        vals = data[[col]].values.astype(np.float32)
        
        if col in to_standardize:
            if col not in new_scalers_x:
                new_scalers_x[col] = StandardScaler()
                vals_proc = new_scalers_x[col].fit_transform(vals)
            else:
                vals_proc = new_scalers_x[col].transform(vals)
            X_dict[arg_name] = torch.tensor(vals_proc.squeeze(), dtype=torch.float32)
        else:
            X_dict[arg_name] = torch.tensor(vals.squeeze(), dtype=torch.float32)
            
    # Target (rating)
    y_vals = data[[target]].values.astype(np.float32)
    if scaler_y is None:
        scaler_y = StandardScaler()
        y_proc = scaler_y.fit_transform(y_vals)
    else:
        y_proc = scaler_y.transform(y_vals)
    
    X_dict[target] = torch.tensor(y_proc.squeeze(), dtype=torch.float32)
    
    return X_dict, new_scalers_x, scaler_y

def plot_predictions(y_true, y_pred, y_std, model_name):
    """
    Plots Predicted vs Actual with uncertainty error bars.
    """
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
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not configured in MODEL_CONFIGS.")
    
    config = MODEL_CONFIGS[model_name]
    model_fn = config["model_fn"]
    pos_filter = config["pos_filter"]
    target_key = config["target"]

    print(f"--- Evaluating {model_name.capitalize()} Model ---")
    
    df = load_PL_dataset()
    pos_df = df[df['position'].str.contains(pos_filter, case=False, na=False)].copy()
    
    train_df, test_df = train_test_split(pos_df, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    train_data, scalers_x, scaler_y = get_model_data(train_df, config)
    test_data, _, _ = get_model_data(test_df, config, scalers_x=scalers_x, scaler_y=scaler_y)

    pyro.clear_param_store()
    guide = AutoNormal(model_fn)
    optimizer = Adam({"lr": lr})
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    print(f"Training on {len(train_df)} samples...")
    for step in range(num_steps):
        loss = svi.step(**train_data)
        if step % 500 == 0:
            print(f"Step {step:4d} : Loss = {loss:.4f}")

    print("\nGenerating predictions on test set...")
    predict_data = {k: v for k, v in test_data.items() if k != target_key}
    
    predictive = Predictive(model_fn, guide=guide, num_samples=500)
    test_samples = predictive(**predict_data)
    
    # error handling since we all used different site names for the rating predictions
    if target_key in test_samples:
        pred_site = target_key
    elif 'obs' in test_samples:
        pred_site = 'obs'
    else:
        # Fallback: look for the one that matches shape
        possible = [k for k, v in test_samples.items() if v.shape[-1] == len(test_df)]
        if possible:
            pred_site = possible[0]
        else:
            raise ValueError(f"Could not find rating predictions in samples. Available: {list(test_samples.keys())}")

    y_pred = test_samples[pred_site].mean(axis=0).detach().numpy()
    y_std = test_samples[pred_site].std(axis=0).detach().numpy()
    y_true = test_data[target_key].detach().numpy()

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
    parser.add_argument("model", choices=["attacker", "goalkeeper", "midfielder", "defender", "combined_manual"], help="Name of the model to evaluate.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of SVI steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    
    args = parser.parse_args()
    
    run_evaluation(args.model, num_steps=args.steps, lr=args.lr)
