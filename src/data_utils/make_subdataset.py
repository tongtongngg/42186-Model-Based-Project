import os

from src.data_utils.load_dataset import load_PL_dataset

def make_subdataset(position='D', output_filename=None):
    """Helper to split data-set by position."""
    df = load_PL_dataset()
    
    subset = df[df['position'] == position]
    
    if output_filename is None:
        output_filename = f"{position}.csv"
    
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', output_filename)
    subset.to_csv(output_path, index=False)
    
    print(f"{position} subset created with {len(subset)} players")
    print(f"Saved to: {output_path}")
    
    return subset

if __name__ == "__main__":
    positions = ['G', 'D', 'M', 'F']
    names = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
    
    for pos, name in zip(positions, names):
        make_subdataset(pos, f"{name}.csv")