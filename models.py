from abc import ABC, abstractmethod
import os
import pickle
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


class UnifiedModelInterface(ABC):

    @abstractmethod
    def fit(self, x_train, y_train, x_val, y_val):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @abstractmethod
    def predict_proba(self, x_test):
        pass

    @abstractmethod
    def save(self, fold_dir):
        pass

    @abstractmethod
    def train_end(self):
        pass


MAP_SKLEARN_MODEL_NAME_CLASS = {('LR', 'binary'): LogisticRegression,
                                ('LR', 'regression'): ElasticNet,
                                ('LR', 'multiclass'): LogisticRegression,
                                ('RF', 'binary'): RandomForestClassifier,
                                ('RF', 'regression'): RandomForestRegressor,
                                ('RF', 'multiclass'): RandomForestClassifier}


class SklearnModel(UnifiedModelInterface):
    def __init__(self, objective, model_name, class_num, **kwargs):
        self.model = MAP_SKLEARN_MODEL_NAME_CLASS[(model_name, objective)](**kwargs)
        self.objective = objective
        self.class_num = class_num

    def fit(self, x_train, y_train, x_val, y_val):
        return self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x):
        if self.objective == 'binary':
            return self.model.predict_proba(x)[:, 1]
        else:
            raise NotImplementedError

    def save(self, fold_dir):
        model_filename = os.path.join(fold_dir, 'sklearn_model.pkl')
        with open(model_filename, 'wb') as file:
            pickle.dump(self.model, file)

    def train_end(self):
        del self.model


MAP_LGBM_MODEL_CLASS = {'binary': LGBMClassifier,
                        'regression': LGBMRegressor}


class LightGBM(UnifiedModelInterface):
    def __init__(self, objective, class_num, **kwargs):
        self.objective = objective
        self.model = MAP_LGBM_MODEL_CLASS[objective](**kwargs)
        self.class_num = class_num

    def fit(self, x_train, y_train, x_val, y_val):
        return self.model.fit(x_train, y_train,
                              eval_set=[(x_val, y_val)],
                              early_stopping_rounds=10)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        if self.objective == 'binary':
            return self.model.predict_proba(x_test)[:, 1]
        else:
            raise NotImplementedError

    def save(self, fold_dir):
        model_filename = os.path.join(fold_dir, 'lgbm_model.txt')
        with open(model_filename, 'wb') as file:
            self.model.booster_.save_model(file)

    def train_end(self):
        del self.model


MAP_XGB_MODEL_CLASS = {'binary': XGBClassifier,
                       'regression': XGBRegressor}


class XGBoost(UnifiedModelInterface):
    def __init__(self, objective, class_num, **kwargs):
        self.model = MAP_XGB_MODEL_CLASS[objective](**kwargs)
        self.objective = objective
        self.class_num = class_num

    def fit(self, x_train, y_train, x_val, y_val):
        return self.model.fit(x_train, y_train,
                              eval_set=[(x_val, y_val)],
                              early_stopping_rounds=10)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        if self.objective == 'binary':
            return self.model.predict_proba(x_test)[:, 1]
        else:
            raise NotImplementedError

    def save(self, fold_dir):
        model_filename = os.path.join(fold_dir, 'xgb_model.txt')
        with open(model_filename, 'wb') as file:
            self.model.save_model(file)

    def train_end(self):
        del self.model


MAR_MLP_MODEL_CLASS = {'binary': MLPClassifier,
                       'regression': MLPRegressor}


class MLP(UnifiedModelInterface):
    def __init__(self, objective, class_num, **kwargs):
        self.model = MAR_MLP_MODEL_CLASS[objective](**kwargs)
        self.objective = objective
        self.class_num = class_num

    def fit(self, x_train, y_train, x_val, y_val):
        shape = x_train.shape[1]
        i = 0
        while(2**i < shape):
            i += 1
        st = i//2
        self.model.set_params(hidden_layer_sizes=(2**st))
        return self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        if self.objective == 'binary':
            return self.model.predict_proba(x_test)[:, 1]
        else:
            raise NotImplementedError

    def save(self, fold_dir):
        model_filename = os.path.join(fold_dir, 'mlp_model.pkl')
        with open(model_filename, 'wb') as file:
            pickle.dump(self.model, file)

    def train_end(self):
        del self.model
