"""Ordinary logistic fits and forests using genotype/time only."""
import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from single_condition_data import HOURS,GENOTYPES


def logistic_curve(hours,r,K,fraction):
    return K/(1+(1/fraction-1)*np.exp(-r*np.asarray(hours)))


def fit_logistic(batch):
    pred=np.empty((5,len(HOURS))); parameters=[]
    frame=batch['means']; target=batch['target'].numpy(); case=batch['case'].numpy()
    for i,g in enumerate(GENOTYPES):
        selected=case==i; hours=frame.elapsed_hours.to_numpy()[selected]; y=target[selected]
        def residual(p): return logistic_curve(hours,*p)-y
        best=None
        for rate in [.025,.05,.1,.2]:
            for factor in [1.1,1.6]:
                capacity=np.clip(float(y.max())*factor,1e-3,1.9)
                start=[rate,capacity,float(np.clip(y[hours.argmin()]/capacity,.001,.9))]
                fit=least_squares(residual,start,bounds=([1e-5,1e-4,1e-5],[.5,2.,.99999]),
                    max_nfev=5000,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                if best is None or np.sum(fit.fun**2)<np.sum(best.fun**2): best=fit
        pred[i]=logistic_curve(HOURS,*best.x)
        parameters.append(dict(genotype=g,r_per_hour=float(best.x[0]),K_normalized=float(best.x[1]),
            initial_fraction=float(best.x[2]),success=bool(best.success),cost=float(np.sum(best.fun**2))))
    return pred,parameters


def forest_features():
    return np.asarray([np.r_[np.eye(5)[i],hour/72.] for i in range(5) for hour in HOURS])


def fit_forest(batch,seed,leaf):
    features=forest_features(); indices=batch['case'].numpy()*len(HOURS)+batch['time'].numpy()
    model=RandomForestRegressor(n_estimators=300,min_samples_leaf=leaf,random_state=seed,n_jobs=2)
    model.fit(features[indices],batch['target'].numpy())
    return model.predict(features).reshape(5,len(HOURS)),model
