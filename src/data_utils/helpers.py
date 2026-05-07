import torch
def standardize(tensor):
    """Standardizes a tensor to have mean 0 and standard deviation 1."""
    return (tensor - tensor.mean()) / tensor.std()