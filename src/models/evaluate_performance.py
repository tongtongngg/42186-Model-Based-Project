import os
import argparse
import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from src.data_utils import load_PL_dataset
from src.models.config import MODEL_CONFIGS

def get_model_data(df, config, scalers_x=None, scaler_y=None):
    """General preprocessing function to prepare data correctly for all models."""
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
            
    y_vals = data[[target]].values.astype(np.float32)
    if scaler_y is None:
        scaler_y = StandardScaler()
        y_proc = scaler_y.fit_transform(y_vals)
    else:
        y_proc = scaler_y.transform(y_vals)
    
    X_dict[target] = torch.tensor(y_proc.squeeze(), dtype=torch.float32)
    
    return X_dict, new_scalers_x, scaler_y

def plot_predictions(y_true, y_pred, y_std, model_name):
    """Plots predicted standardized ratings vs actual standardized ratings with error bars. 
    Red line indicates perfect predictions (congruence)."""
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
    """General inference and evaluation loop for any model defined in MODEL_CONFIGS."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not configured in MODEL_CONFIGS.")
    
    config = MODEL_CONFIGS[model_name]
    model_fn = config["model_fn"]
    
    df = load_PL_dataset()

    if config.get("is_combined_all"):
        train_data = {}
        test_data_full = {}
        scalers = {}
        
        all_y_true = []
        all_y_pred = []
        all_y_std = []

        for pos_key, pos_cfg in config["positions"].items():
            pos_df = df[df['position'].str.contains(pos_cfg['pos_filter'], case=False, na=False)].copy()
            tr_df, te_df = train_test_split(pos_df, test_size=0.2, random_state=42)
            
            tr_data, sx, sy = get_model_data(tr_df, pos_cfg)
            te_data, _, _ = get_model_data(te_df, pos_cfg, scalers_x=sx, scaler_y=sy)
            
            train_data[f"{pos_key}_data"] = tr_data
            test_data_full[f"{pos_key}_data"] = te_data
            scalers[pos_key] = {"sx": sx, "sy": sy, "te_len": len(te_df)}

        pyro.clear_param_store()
        guide = AutoNormal(model_fn)
        svi = SVI(model_fn, guide, Adam({"lr": lr}), loss=Trace_ELBO())

        for step in range(num_steps):
            loss = svi.step(**train_data)
            if step % 500 == 0: print(f"Step {step:4d} : Loss = {loss:.4f}")

        predict_data = {pk: {k: v for k, v in pd_val.items() if k != "rating"} for pk, pd_val in test_data_full.items()}
        predictive = Predictive(model_fn, guide=guide, num_samples=500)
        samples = predictive(**predict_data)

        for pos_key, pos_cfg in config["positions"].items():
            site = pos_cfg["target_site"]
            y_pred = samples[site].mean(axis=0).detach().numpy()
            y_std = samples[site].std(axis=0).detach().numpy()
            y_true = test_data_full[f"{pos_key}_data"]["rating"].detach().numpy()
            
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            all_y_std.extend(y_std)

        all_y_true, all_y_pred, all_y_std = np.array(all_y_true), np.array(all_y_pred), np.array(all_y_std)
        
        mse = mean_squared_error(all_y_true, all_y_pred)
        mae = mean_absolute_error(all_y_true, all_y_pred)
        r2 = r2_score(all_y_true, all_y_pred)

        print(f"\n==============================\n      {model_name.upper()} PERFORMANCE\n==============================\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR2 : {r2:.4f}\n==============================")
        plot_predictions(all_y_true, all_y_pred, all_y_std, model_name)

    else:
        pos_filter = config["pos_filter"]
        target_key = config["target"]
        pos_df = df[df['position'].str.contains(pos_filter, case=False, na=False)].copy()
        train_df, test_df = train_test_split(pos_df, test_size=0.2, random_state=42)
        
        train_data, sx, sy = get_model_data(train_df, config)
        test_data, _, _ = get_model_data(test_df, config, scalers_x=sx, scaler_y=sy)

        pyro.clear_param_store()
        guide = AutoNormal(model_fn)
        svi = SVI(model_fn, guide, Adam({"lr": lr}), loss=Trace_ELBO())

        for step in range(num_steps):
            loss = svi.step(**train_data)
            if step % 500 == 0: print(f"Step {step:4d} : Loss = {loss:.4f}")

        predict_data = {k: v for k, v in test_data.items() if k != target_key}
        predictive = Predictive(model_fn, guide=guide, num_samples=500)
        test_samples = predictive(**predict_data)
        
        pred_site = target_key if target_key in test_samples else 'obs' if 'obs' in test_samples else [k for k, v in test_samples.items() if v.shape[-1] == len(test_df)][0]

        y_pred = test_samples[pred_site].mean(axis=0).detach().numpy()
        y_std = test_samples[pred_site].std(axis=0).detach().numpy()
        y_true = test_data[target_key].detach().numpy()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"\n==============================\n      {model_name.upper()} PERFORMANCE\n==============================\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR2 : {r2:.4f}\n==============================")
        plot_predictions(y_true, y_pred, y_std, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate performance of MBML models.")
    parser.add_argument("model", choices=list(MODEL_CONFIGS.keys()), help="Name of the model to evaluate.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of SVI steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    args = parser.parse_args()
    run_evaluation(args.model, num_steps=args.steps, lr=args.lr)
