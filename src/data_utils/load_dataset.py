import glob
import os

import kagglehub
import pandas as pd

def load_PL_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download("sananmuzaffarov/epl-202526-player-stats-gw131")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.read_csv(csv_files[0], encoding='latin-1')

    return df

if __name__ == "__main__":
    df = load_PL_dataset()

    print(df.head(517))