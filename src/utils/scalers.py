from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
class BandwidthScaler(BaseEstimator, TransformerMixin):
    """
    Inherit get_params() and set_params() for BaseEstimator and 
    fit_transform() from TransformerMixin
    """
    def __init__(self, bandwidth: float = 1.0):
        super().__init__()
        self.bandwidth = bandwidth
        self.standard_scaler = StandardScaler()

    def fit(self, x, y=None):
        self.standard_scaler.fit(x)
        return self
    
    def transform(self, x):
        x_scaled = self.standard_scaler.transform(x)
        x_scaled *= self.bandwidth
        return x_scaled
    
class CustomLabelMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def fit(self, y):
        self.scaler.fit(y.reshape(-1,1))
        return self

    def transform(self, y):
        return self.scaler.transform(y.reshape(-1,1)).ravel()

    def inverse_transform(self, y):
        # skip inverse transformation by returning the input unchanged
        return y
    
class LabelMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range: tuple[float, float] = (0.0,1.0)):
        super().__init__()
        self.feature_range = feature_range
        self.min_max_scaler = MinMaxScaler(self.feature_range)

    def fit(self, y):
        self.min_max_scaler.fit(y)
        return self
    
    def transform(self, y):
        y_scaled = self.min_max_scaler.transform(y)
        return y_scaled