# 1. Imports
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import genpareto, laplace
from scipy.optimize import minimize
import dill

# 2. Utility functions
def fit_gpd(data, percentile):
    """Fit GPD to data above a percentile threshold"""
    threshold = np.percentile(data, percentile)
    exceed = data[data > threshold] - threshold
    shape, loc, scale = genpareto.fit(exceed, floc=0)
    return dict(shape=shape, scale=scale, threshold=threshold)

def to_laplace(data):
    """Transform to Laplace space"""
    ranks = (np.argsort(np.argsort(data)) + 1) / (len(data) + 1)
    return laplace.ppf(ranks)

def from_laplace(z):
    """Inverse Laplace transform (approx)"""
    ranks = laplace.cdf(z)
    return np.quantile(z, ranks)
  
# 3. Core modeling class
class MSTMTEModel:

    def __init__(self, region="guadeloupe-wide", depth=-100,
                 thr_mar=0.6, thr_com=0.9):
        self.region = region
        self.depth = depth
        self.thr_mar = thr_mar
        self.thr_com = thr_com
        self.fitted = False

    # Data preparation
    def load_data(self, nc_path):
        """Load and filter raw .nc files"""
        ds = xr.open_dataset(nc_path)
        # Add subsetting logic here
        self.data = ds

    # Model fitting
    def fit(self):
        """Fit marginal GPDs and conditional Heffernan-Tawn model"""
        # Example using Hs, U variables
        self.params = {}
        for var in ["Hs", "U"]:
            self.params[var] = fit_gpd(self.data[var].values, self.thr_mar*100)
        self.fitted = True

    # Simulation
    def simulate(self, return_period=100):
        """Generate synthetic extremes using conditional Heffernan-Tawn"""
        if not self.fitted:
            raise RuntimeError("Run .fit() first.")
        # Insert conditional extremes logic here
        # Placeholder simulation
        self.simulated = {k: np.random.randn(1000) for k in self.params.keys()}

    # Save / Load
    def save(self, path="mstme_condition.dill"):
        """Save model to file"""
        with open(path, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(path):
        """Load existing model"""
        with open(path, "rb") as f:
            return dill.load(f)


# 4. Visualization utilities
def plot_return_curves(model):
    """Plot simulated vs observed extremes"""
    plt.figure(figsize=(5,4))
    for var, sim in model.simulated.items():
        plt.hist(sim, bins=30, alpha=0.5, label=var)
    plt.legend(); plt.xlabel("Value"); plt.ylabel("Frequency")
    plt.title("Simulated Extreme Distributions")
    plt.show()


# 5. Command line interface
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MSTM-TE framework")
    parser.add_argument("--fit", type=str, help="Path to .nc data to fit")
    parser.add_argument("--simulate", type=int, help="Return period to simulate")
    args = parser.parse_args()

    model = MSTMEModel()

    if args.fit:
        model.load_data(args.fit)
        model.fit()
        model.save()

    if args.simulate:
        model = MSTMEModel.load("mstme_condition.dill")
        model.simulate(return_period=args.simulate)
        plot_return_curves(model)
