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
def gp_cdf(mu, sigma, kesi, x1, x2):
    """
    Compute the cumulative distribution function of the Generalized pareto distribution
    :param mu: location parameter
    :param sigma: scale parameter
    :param kesi: shape parameter
    :param x1: xi − δ ( refer the paper "Extreme value estimation using the likelihood-weighted method"
    :param x2: xi + δ ( refer the paper "Extreme value estimation using the likelihood-weighted method"
    :return: the cumulative distribution function
    """
    condition_1 = kesi >= 0
    condition_2 = x1 - mu <= -sigma / kesi
    condition_3 = x2 - mu <= -sigma / kesi
    condition = np.logical_and(np.logical_or(condition_1, condition_2), np.logical_or(condition_1, condition_3))
    cdf_1 = 1 - (1 + kesi * (x1 - mu) / sigma) ** (-1 / kesi * condition)
    cdf_2 = 1 - (1 + kesi * (x2 - mu) / sigma) ** (-1 / kesi * condition)
    cdf = cdf_2 - cdf_1
    return cdf


def likelihood_weight(data, threshold_value, sigma_upper=100):
    """
    Compute the parameters of GP distribution using likelihood-weighted method
    refer paper :"Extreme value estimation using the likelihood-weighted method"
    :param data: input observation data
    :param threshold_value: given high threshold value
    :param sigma_upper: upper bound of sigma
    :param sigma_num: the number of sigma divided
    :return: scale and shape parameters
    """
    pot_value = data
    sigma = np.arange(0.01, sigma_upper, 0.1)
    kesi = np.linspace(-0.999999, 0.9999, 100)
    sigma_m, kesi_m = np.meshgrid(sigma, kesi)
    cdf_pro = np.ones_like(sigma_m)
    for item in pot_value:
        cdf = gp_cdf(threshold_value, sigma_m, kesi_m, item - 0.005, item + 0.005)
        cdf_pro = cdf_pro * cdf
        cdf_pro = cdf_pro / np.sum(cdf_pro)

    max_position = np.unravel_index(cdf_pro.argmax(), cdf_pro.shape)
    sigma = sigma_m[max_position[0]][max_position[1]]
    kesi = kesi_m[max_position[0]][max_position[1]]
    return sigma, kesi


# 2. Utility functions
def fit_gpd(data, percentile):
    """Fit GPD to data above a percentile threshold"""
    threshold = np.percentile(data, percentile)
    exceed = data[data > threshold] - threshold
    scale, shape = likelihood_weight(exceed, threshold)  # Compute the parameters of GP distribution using likelihood-weighted method
    return dict(shape=shape, scale=scale, threshold=threshold, percentile=percentile)


def gdp_cdf(x, threshold, shape, scale, percentile):
    """the cumulative distribution function"""
    # p_cumulative = percentile + (1 - percentile) * genpareto.cdf(x,loc=threshold, c=shape, scale=scale)
    p_cumulative = percentile + (1 - percentile) * genpareto.cdf(x, loc=threshold, c=shape, scale=scale)
    return p_cumulative


def gpd_ppf(p_cumulative, threshold, shape, scale, percentile):
    """Percent Point Function, inverse of the cumulative distribution function"""
    # p_cumulative = percentile + (1 - percentile) * genpareto.cdf(x,loc=threshold, c=shape, scale=scale)
    # (p_cumulative - percentile) / (1 - percentile) = genpareto.cdf(x,loc=threshold, c=shape, scale=scale)
    p_cumulative_transform = (p_cumulative - percentile) / (1 - percentile)
    return genpareto.ppf(p_cumulative_transform, loc=threshold, c=shape, scale=scale)


def to_laplace(p_cumulative):
    """Transform to Laplace space
        p_cumulative: Cumulative probability
    """
    p_cumulative = np.array(p_cumulative)
    data_laplace = np.where(
        p_cumulative < 0.5,
        np.log(2 * p_cumulative),
        -np.log(2 * (1 - p_cumulative))
    )
    return data_laplace


def from_laplace(data_laplace):
    """Inverse Laplace transform
    data_laplace: Samples in Laplace space
    p_cumulative: Cumulative probability
    """
    data_laplace = np.array(data_laplace)
    p_cumulative = np.where(
        data_laplace >= 0,
        1 - 0.5 * np.exp(-data_laplace),
        0.5 * np.exp(data_laplace)
    )
    return p_cumulative

  
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
        self.form_partitions(percentile=self.thr_mar * 100)
        self.fitted = True
        
    # Partition Formation
    def form_partitions(self, percentile=70):
        """Form partitions C_d where each variable is the driver of the joint extreme"""
        df = pd.DataFrame({
            "Hs": self.data["Hs"].values,
            "U": self.data["U"].values
        })
        # Thresholds ψ_d
        psi = {col: np.percentile(df[col], percentile) for col in df.columns}
        # Identify rows where variable exceeds its threshold and is the maximum
        partitions = {col: df.index[(df[col] > psi[col]) & (df[col] == df.max(axis=1))].tolist()
                      for col in df.columns}
        # Empirical frequencies ρ_d
        counts = {col: len(partitions[col]) for col in df.columns}
        total = sum(counts.values()) or 1
        rho = {col: counts[col] / total for col in df.columns}
        self.partitions, self.rho, self.psi = partitions, rho, psi
    
    # Simulation
    def simulate(self, return_period=100):
        """Generate synthetic extremes using conditional Heffernan-Tawn"""
        if not self.fitted:
            raise RuntimeError("Run .fit() first.")
        # Insert conditional extremes logic here
        # Placeholder simulation
        self.simulated = {k: np.random.randn(1000) for k in self.params.keys()}
        
    # Attachment of time exposure
    def attach_time_exposure(self, stm_values, exposure_array):
        """Attach time-exposure to simulated values"""
        tc_series = []
        for val in stm_values:
            # Randomly choose an exposure pattern and scale it by the STM magnitude
            exposure = exposure_array[np.random.randint(0, len(exposure_array))]
            tc_series.append(val * exposure)
        return tc_series

    # Computation of return value
    def compute_return_value(data, return_period, annual_frequency=1):
        """Return level corresponding to given return period"""
        p = 1 - 1 / (return_period * annual_frequency)
        return np.quantile(data, p)

    # Save / Load
    def save(self, path="mstmte_condition.dill"):
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
        model = MSTMEModel.load("mstmte_condition.dill")
        model.simulate(return_period=args.simulate)
        plot_return_curves(model)
