import pandas as pd
import os

# Path to the defender CSV
csv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'Defender.csv')

# Load the defender dataset
df = pd.read_csv(csv_path, encoding='utf-8')

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Compute Spearman correlations with 'rating'
correlations = numeric_df.corr(method='spearman')['rating']

# Drop the 'rating' itself
correlations = correlations.drop('rating', errors='ignore')

# Sort by absolute correlation value
correlations = correlations.sort_values(key=abs, ascending=False)

# Create a DataFrame for saving
corr_df = correlations.reset_index()
corr_df.columns = ['parameter', 'correlation']

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'defender_correlations.csv')
corr_df.to_csv(output_path, index=False)

print(f"Defender correlations saved to: {output_path}")
print("Top 10 correlations:")
print(corr_df.head(10))