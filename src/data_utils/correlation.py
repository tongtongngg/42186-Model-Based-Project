import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.data_utils.load_dataset import load_PL_dataset

def get_correlations_by_position(df, target_col='rating', positions=['F', 'M', 'D', 'G'], cutoff=0.6):

    pos_correlation_map = {}

    for pos in positions:
        pos_df = df[df['position'].str.upper() == pos.upper()].copy()
        
        if pos_df.empty:
            pos_correlation_map[pos] = "No data found for this position"
            continue

        numeric_df = pos_df.select_dtypes(include=['number'])
        
        # Using Spearman because football stats are often skewed
        correlations = numeric_df.corr(method='spearman')[target_col]
        
        correlations = correlations.drop(labels=[target_col], errors='ignore')
        
        filtered_corr = correlations[correlations.abs() >= cutoff]

        sorted_corr = filtered_corr.sort_values(key=abs, ascending=False)        
        pos_correlation_map[pos] = sorted_corr.to_dict()

    return pos_correlation_map


if __name__ == "__main__":
    data = load_PL_dataset()
    results = get_correlations_by_position(data, positions=['F'])
    print(list(results['F'].items()))
    print(len(results['F']))