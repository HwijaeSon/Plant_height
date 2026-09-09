"""Fit process models and random forests directly to every measured training cell."""
import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from data import CASES, HOURS
from process_baselines import logistic_curve, forest_features


def fit_forest(batch, seed, leaf):
    case, time = [batch[k].cpu().numpy() for k in ['case', 'time']]
    target = batch['length'].cpu().numpy()
    features = forest_features()
    indices = case*len(HOURS) + time
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf,
                                  random_state=seed, n_jobs=2)
    model.fit(features[indices], target)
    return model.predict(features).reshape(len(CASES), len(HOURS)), model


def fit_logistic(batch, switched):
    case, time = [batch[k].cpu().numpy() for k in ['case', 'time']]
    target = batch['length'].cpu().numpy()
    predictions = np.zeros((len(CASES), len(HOURS)))
    records = []
    for group in range(5 if switched else 10):
        indices = [2*group, 2*group+1] if switched else [group]
        selected = np.isin(case, indices)
        selected_target = target[selected]
        local_case = np.array([indices.index(int(i)) for i in case[selected]])
        selected_time = time[selected]
        maximum = float(selected_target.max())
        if switched:
            bounds = ([1e-5, 1e-5, 1e-4, 1e-5, 1e-5], [.5, .5, 2., .99999, .99999])
            def predict(p):
                return np.stack([logistic_curve(HOURS, p[0], p[1], p[2], p[3+j], CASES[i][1])
                                 for j, i in enumerate(indices)])
        else:
            bounds = ([1e-5, 1e-4, 1e-5], [.5, 2., .99999])
            def predict(p):
                return logistic_curve(HOURS, p[0], p[0], p[1], p[2], CASES[indices[0]][1])[None]
        def residual(p):
            return predict(p)[local_case, selected_time] - selected_target
        best = None
        for rate in [.025, .05, .1, .2]:
            for factor in [1.1, 1.6]:
                capacity = min(maximum*factor, 1.9)
                # A measured initial value supplies a numerical starting point only.
                fractions = [np.clip(float(target[(case == i) & (time == 0)][0])/capacity, .001, .9)
                             for i in indices]
                start = [rate, rate, capacity, *fractions] if switched else [rate, capacity, *fractions]
                result = least_squares(residual, start, bounds=bounds, max_nfev=5000,
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10)
                if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
                    best = result
        predictions[indices] = predict(best.x)
        records.append(dict(cases=indices, parameters=best.x.tolist(), success=bool(best.success),
                            cost=float(np.sum(best.fun**2)), n_training_measurements=int(selected.sum())))
    return predictions, records
