import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
import pandas as pd
from src.data_utils import load_PL_dataset

def standardize(tensor):
    """Standardizes a tensor to have mean 0 and standard deviation 1."""
    return (tensor - tensor.mean()) / tensor.std()

def HierarchicalManualModel():
    """
    A simple Hierarchical Bayesian Regression model using the manually chosen parameters from our individual models.
    """
    