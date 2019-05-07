#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd


# In[2]:


class Preprocessor:
    def __init__(self, scaling = None, split = None, n_splits = 2):
        self.scaling = scaling
        self.split = split
        self.n_splits = n_splits
        
    def get_scaled(self, X, X_test=None):
        if self.scaling == 'MinMax':
            scaler = MinMaxScaler()
        elif self.scaling == 'Standard':
            scaler = StandardScaler()
        if X_test is not None:
            return pd.DataFrame(scaler.fit_transform(X), index = X.index, columns = X.columns), pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
        else:
            return pd.DataFrame(scaler.fit_transform(X), index = X.index, columns = X.columns)
    
    def get_split(self, X, y=None):
        if self.split == 'Kfold':
            kf = KFold(n_splits = self.n_splits)
        elif self.split == 'StratifiedKFold':
            kf = StratifiedKFold(n_splits = self.n_splits)
        if y is not None:
            return list(kf.split(X, y))
        else:
            return list(kf.split(X))


# In[ ]:




