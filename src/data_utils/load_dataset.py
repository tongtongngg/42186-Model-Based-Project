import glob
import os

import kagglehub
import pandas as pd

def load_PL_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download("sananmuzaffarov/epl-202526-player-stats-gw131")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.read_csv(csv_files[0], encoding='latin-1')
    
    # Save a copy to project folder for easy access
    local_csv = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'epl_player_stats.csv')
    os.makedirs(os.path.dirname(local_csv), exist_ok=True)
    df.to_csv(local_csv, index=False)
    print(f"Data saved to: {local_csv}")

    return df

if __name__ == "__main__":
    df = load_PL_dataset()

    print(df.head(517))