"""Exact ordinary and light-switched logistic solutions fitted to training means."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor

from data import CASES, HOURS, light, light_exposure


def logistic_curve(hours, rate_light, rate_dark, capacity, initial_fraction, condition):
    exposure = light_exposure(hours, condition)
    accumulated = rate_light * exposure + rate_dark * (np.asarray(hours)-exposure)
    return capacity / (1 + (1/initial_fraction-1)*np.exp(-accumulated))


def fit_logistic(batch, light_switched):
    target = batch["target"].numpy()
    mask = batch["mask"].numpy().astype(bool)
    prediction = np.zeros_like(target)
    parameters = []
    for group in range(5 if light_switched else 10):
        indices = [2*group, 2*group+1] if light_switched else [group]
        maximum = float(target[indices].max())
        best = None
        if light_switched:
            bounds = ([1e-5, 1e-5, 1e-4, 1e-5, 1e-5], [.5, .5, 2., .99999, .99999])
            def predict(p):
                return np.stack([logistic_curve(HOURS, p[0], p[1], p[2], p[3+j], CASES[i][1])
                                 for j, i in enumerate(indices)])
        else:
            bounds = ([1e-5, 1e-4, 1e-5], [.5, 2., .99999])
            def predict(p):
                return logistic_curve(HOURS, p[0], p[0], p[1], p[2], CASES[indices[0]][1])[None]
        def residual(p):
            return (predict(p)-target[indices])[mask[indices]]
        for rate in (.025, .05, .1, .2):
            for factor in (1.1, 1.6):
                K = min(maximum*factor, 1.9)
                fractions = [np.clip(float(target[i,0])/K, .001, .9) for i in indices]
                start = [rate, rate, K, *fractions] if light_switched else [rate, K, *fractions]
                result = least_squares(residual, start, bounds=bounds, max_nfev=5000,
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10)
                if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
                    best = result
        prediction[indices] = predict(best.x)
        parameters.append(dict(cases=indices, fitted=best.x.tolist(), success=bool(best.success),
                               cost=float(np.sum(best.fun**2))))
    return prediction, parameters


def forest_features():
    rows=[]
    for i,(genotype,condition) in enumerate(CASES):
        encoding=np.eye(5)[i//2]
        for hour in HOURS:
            rows.append(np.r_[encoding, hour/72., light(hour,condition),
                               .5 if condition == "12L12D" else 1., light_exposure(hour,condition)/72.])
    return np.asarray(rows)


def fit_forest(batch, seed, min_samples_leaf):
    features=forest_features()
    mask=batch["mask"].numpy().astype(bool).ravel()
    targets=batch["target"].numpy().ravel()
    model=RandomForestRegressor(n_estimators=300, min_samples_leaf=min_samples_leaf,
                               random_state=seed, n_jobs=2)
    model.fit(features[mask],targets[mask])
    return model.predict(features).reshape(10,len(HOURS)),model
