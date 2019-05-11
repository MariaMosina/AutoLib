from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd


SCALER = {'MinMax': MinMaxScaler,
          'Standard': StandardScaler}
SPLITTER = {'KFold': KFold,
            'StratifiedKFold': StratifiedKFold}


class Preprocessor:
    def __init__(self, scaling=None, split=None, n_splits=2):
        self.scaling = scaling
        self.split = split
        self.n_splits = n_splits

    def get_scaled(self, x, x_test=None):
        if self.scaling is not None:
            scaler = SCALER[self.scaling]()
            if x_test is not None:
                return pd.DataFrame(scaler.fit_transform(x),
                                    index=x.index,
                                    columns=x.columns), pd.DataFrame(scaler.transform(x_test),
                                                                    index=x_test.index,
                                                                    columns=x_test.columns)
            else:
                return pd.DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns)

    def get_split(self, x, y=None):
        if self.split is not None:
            kf = SPLITTER[self.split](n_splits=self.n_splits)
            if y is not None:
                return list(kf.split(x, y))
            else:
                return list(kf.split(x))
