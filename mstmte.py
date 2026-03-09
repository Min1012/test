#imports
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from scipy.optimize import minimize

#laplace transform
def to_laplace(p):

    p = np.asarray(p)

    return np.where(
        p < 0.5,
        np.log(2*p),
        -np.log(2*(1-p))
    )

#inverse laplace
def from_laplace(x):

    x = np.asarray(x)

    return np.where(
        x >= 0,
        1 - 0.5*np.exp(-x),
        0.5*np.exp(x)
    )

#GPD fitting
def fit_gpd(data, threshold_quantile):

    threshold = np.quantile(data, threshold_quantile)

    exceed = data[data > threshold] - threshold

    c, loc, scale = genpareto.fit(exceed)

    def cdf(x):

        p = np.where(
            x <= threshold,
            threshold_quantile,
            threshold_quantile +
            (1 - threshold_quantile)
            * genpareto.cdf(x-threshold, c, scale=scale)
        )

        return p


    def ppf(p):

        p2 = (p - threshold_quantile)/(1-threshold_quantile)

        return threshold + genpareto.ppf(p2, c, scale=scale)

    return {
        "threshold":threshold,
        "shape":c,
        "scale":scale,
        "cdf":cdf,
        "ppf":ppf
    }

#extract mstm
def extract_mstm(df):

    variables = [c for c in df.columns if c not in ["storm_id","time"]]

    mstm = {v:[] for v in variables}

    for sid, group in df.groupby("storm_id"):

        for v in variables:

            mstm[v].append(group[v].max())

    return {v:np.array(mstm[v]) for v in variables}

#extract te
def extract_te(df):

    variables = [c for c in df.columns if c not in ["storm_id","time"]]

    exposures = []

    for sid, group in df.groupby("storm_id"):

        te = {}

        for v in variables:

            maxv = group[v].max()

            if maxv == 0:
                te[v] = group[v].values
            else:
                te[v] = group[v].values / maxv

        exposures.append(te)

    return exposures

#fit conditional model
def fit_ht_model(laplace_data):

    variables = list(laplace_data.keys())

    params = {}

    for d in variables:

        y = laplace_data[d]

        for v in variables:

            if v == d:
                continue

            x = laplace_data[v]

            def loss(p):

                a, b = p

                z = (x - a*y)/(np.abs(y)**b)

                return np.var(z)

            result = minimize(loss,[0.5,0.1])

            params[(v,d)] = result.x

    return params

#conditional simulation
def simulate_ht(params):

    variables = list(set([k[0] for k in params] + [k[1] for k in params]))

    d = np.random.choice(variables)

    y = {}

    y[d] = np.random.exponential()

    for v in variables:

        if v == d:
            continue

        a,b = params[(v,d)]

        z = np.random.normal()

        y[v] = a*y[d] + (np.abs(y[d])**b)*z

    return y

#main model
class MSTMTE:

    def __init__(self, threshold=0.6):

        self.threshold = threshold
        self.fitted = False


#fit model
    def fit(self, data):

        if not isinstance(data,pd.DataFrame):
            data = pd.DataFrame(data)

        self.data = data

        self.mstm = extract_mstm(data)

        self.te = extract_te(data)

        self.gpd = {}

        for var in self.mstm:

            self.gpd[var] = fit_gpd(self.mstm[var],self.threshold)

        self.laplace = {}

        for var in self.mstm:

            p = self.gpd[var]["cdf"](self.mstm[var])

            self.laplace[var] = to_laplace(p)

        self.ht = fit_ht_model(self.laplace)

        self.fitted = True


#simulation
    def simulate(self,n=1000):

        if not self.fitted:
            raise RuntimeError("Model not fitted")

        storms = []

        for i in range(n):

            y = simulate_ht(self.ht)

            mstm = {}

            for var in y:

                p = from_laplace(y[var])

                mstm[var] = self.gpd[var]["ppf"](p)

            te = self.te[np.random.randint(len(self.te))]

            storm = {}

            for var in mstm:

                storm[var] = mstm[var] * te[var]

            storms.append(storm)

        return storms


#return levels
    def return_level(self, peaks, T):

        p = 1 - 1/T

        return np.quantile(peaks,p)


