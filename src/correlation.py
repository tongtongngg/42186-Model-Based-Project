import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.data_utils.load_dataset import load_PL_dataset

def get_correlations_by_position(df, target_col='rating', positions=['F', 'M', 'D', 'G'], cutoff=0.7):

    pos_correlation_map = {}

    for pos in positions:
        pos_df = df[df['position'].str.upper() == pos.upper()].copy()
        
        if pos_df.empty:
            pos_correlation_map[pos] = "No data found for this position"
            continue

        # 2. Select only numeric columns for correlation
        numeric_df = pos_df.select_dtypes(include=['number'])
        
        # Using Spearman because football stats are often skewed
        correlations = numeric_df.corr(method='spearman')[target_col]
        
        correlations = correlations.drop(labels=[target_col], errors='ignore')
        
        filtered_corr = correlations[correlations.abs() >= cutoff]

        # Sort by absolute value (highest impact first)
        sorted_corr = filtered_corr.reindex(filtered_corr.abs().sort_values(ascending=False).index)
        
        pos_correlation_map[pos] = sorted_corr.to_dict()

    return pos_correlation_map


if __name__ == "__main__":
    data = load_PL_dataset()
    results = get_correlations_by_position(data)
    print(list(results['M'].items()))