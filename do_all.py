from preprocessor import *
from models import *
import pandas as pd
from sklearn.model_selection import train_test_split


def do_all(train_data, test_data, y=None, scaler=None, splitter=None, objective=None, model=None):
    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    pr = Preprocessor(scaling=scaler)
    train, test = pr.get_scaled(train.drop(y, axis=1), test)
    x_train, x_val, y_train, y_val = train_test_split(train.drop(y, axis=1), train.y,
                                                      test_size=0.33, random_state=42)
    model = SklearnModel(objective=objective, model_name=model)
    model.fit(x_train, y_train, x_val, y_val)
    pred = model.predict(test)
    pred.to_csv('prediction.csv')


do_all()

