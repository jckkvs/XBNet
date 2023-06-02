import torch

from sklearn.base import BaseEstimator, RegressorMixin

from .models import XBNETRegressor as XBNR
from .run import run_XBNET


class XBNETRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, num_layers=2, num_layers_boosted=1):
        self.num_layers = num_layers
        self.num_layers_boosted = num_layers_boosted

    def fit(self, X, y):
        self.base_estimator = XBNR(
            X, y, num_layers=self.num_layers, num_layers_boosted=self.num_layers_boosted
        )
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.base_estimator.parameters(), lr=0.01)
        run_XBNET(X, y, self.base_estimator, criterion, optimizer, 32, 300)

    def predict(self, X):
        return self.base_estimator(X)
