import pandas as pd
import os

# Path to the defender CSV
csv_path = os.path.join(
    os.path.dirname(__file__),
    '..',
    '..',
    '..',
    'data',
    'Defender.csv'
)

# Load the defender dataset
df = pd.read_csv(csv_path, encoding='utf-8')

# Parameters to analyze
parameters = [
    "accuratePasses",
    "touches",
    "totalDuelsWon",
    "clearances",
    "cleanSheet"
]

# Keep only selected columns
selected_df = df[parameters]

# Compute Spearman correlation matrix
correlation_matrix = selected_df.corr(method='spearman')

# Save correlation matrix
output_path = os.path.join(
    os.path.dirname(__file__),
    '..',
    '..',
    '..',
    'data',
    'defender_parameter_correlations.csv'
)

correlation_matrix.to_csv(output_path)

# Print results
print("Spearman Correlation Matrix:")
print(correlation_matrix)

print(f"\nCorrelation matrix saved to: {output_path}")