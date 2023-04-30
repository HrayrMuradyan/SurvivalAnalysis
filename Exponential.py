from lifelines.fitters import ParametricRegressionFitter
from autograd import numpy as np

class ExponentialAFTFitter(ParametricRegressionFitter):

    _fitted_parameter_names = ['lambda_']

    def _cumulative_hazard(self, params, t, Xs):
        beta = params['lambda_']
        X = Xs['lambda_']
        lambda_ = np.exp(np.dot(X, beta))
        return t / lambda_
