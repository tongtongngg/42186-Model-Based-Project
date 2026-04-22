import pandas as pd
import os

def make_subdataset(position='D', output_filename=None):
    """
    Create a subset of the EPL dataset filtered by position.
    
    Args:
        position (str): The position to filter by (e.g., 'M' for midfielders)
        output_filename (str): Name of the output CSV file (optional)
    """
    # Path to the original CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'epl_player_stats.csv')
    
    # Load the dataset
    df = pd.read_csv(csv_path, encoding='latin-1')
    
    # Filter by position
    subset = df[df['position'] == position]
    
    # Default output filename
    if output_filename is None:
        output_filename = f"{position.lower()}_players.csv"
    
    # Save the subset
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', output_filename)
    subset.to_csv(output_path, index=False)
    
    print(f"{position} subset created with {len(subset)} players")
    print(f"Saved to: {output_path}")
    
    return subset

if __name__ == "__main__":
    # Create defenders subset
    make_subdataset('D', 'defenders.csv')