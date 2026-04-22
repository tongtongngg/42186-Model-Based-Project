import pandas as pd
import os
import sys

def make_subdataset(position='D', output_filename=None):

    # Path to the original CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'epl_player_stats.csv')
    
    # Load the dataset
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Filter by position
    subset = df[df['position'] == position]
    
    # Default output filename
    if output_filename is None:
        output_filename = f"{position}.csv"
    
    # Save the subset
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', output_filename)
    subset.to_csv(output_path, index=False)
    
    print(f"{position} subset created with {len(subset)} players")
    print(f"Saved to: {output_path}")
    
    return subset

if __name__ == "__main__":
    # Create subsets for all positions
    positions = ['G', 'D', 'M', 'F']
    names = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
    
    for pos, name in zip(positions, names):
        make_subdataset(pos, f"{name}.csv")