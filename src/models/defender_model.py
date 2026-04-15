import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_utils.load_dataset import load_PL_dataset
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# Load the Premier League dataset
df = load_PL_dataset()

# Filter for defenders (position == "D")
defender_data = df[df['position'] == 'D']

print("Defender data loaded successfully!")
print(f"Dataset shape: {defender_data.shape}")

# Select numerical columns (exclude strings)
numerical_cols = defender_data.select_dtypes(include=['number']).columns.tolist()
print(f"Numerical columns: {len(numerical_cols)}")

# Drop columns with any NaN
defender_clean = defender_data[numerical_cols].dropna(axis=1, how='any')

print(f"Data shape after dropping NaN columns: {defender_clean.shape}")
print(f"Remaining columns: {list(defender_clean.columns)}")

# Separate rating and features
if 'rating' in defender_clean.columns:
    rating = defender_clean['rating']
    features = defender_clean.drop('rating', axis=1)
else:
    print("No 'rating' column found")
    features = defender_clean
    rating = None

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(features_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by component: {explained_variance[:10]}")

# Plot explained variance
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(explained_variance)+1), explained_variance.cumsum(), marker='o')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('PCA Explained Variance')
# plt.grid(True)
# plt.savefig('pca_explained_variance.png')
# print("Saved PCA explained variance plot to pca_explained_variance.png")

print("Cumulative explained variance:")
for i, cum_var in enumerate(explained_variance.cumsum()[:10]):
    print(f"PC{i+1}: {cum_var:.3f}")

# If rating exists, correlate with principal components
if rating is not None:
    # Correlation between rating and first few PCs
    for i in range(min(5, pca_result.shape[1])):
        corr = pd.Series(pca_result[:, i]).corr(rating)
        print(f"Correlation between rating and PC{i+1}: {corr:.3f}")

    # Plot rating vs first PC
    # plt.figure(figsize=(8, 6))
    # plt.scatter(pca_result[:, 0], rating, alpha=0.5)
    # plt.xlabel('First Principal Component')
    # plt.ylabel('Rating')
    # plt.title('Rating vs First Principal Component')
    # plt.grid(True)
    # plt.savefig('rating_vs_pc1.png')
    # print("Saved rating vs PC1 plot to rating_vs_pc1.png")

    print(f"Rating range: {rating.min():.2f} - {rating.max():.2f}")
    print(f"PC1 range: {pca_result[:, 0].min():.2f} - {pca_result[:, 0].max():.2f}")

