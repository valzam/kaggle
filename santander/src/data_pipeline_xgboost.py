import pandas as pd
import pickle

from average_precision import calculate_delta
from create_sequences import create_sequence, y_cols
from preprocess import cleanup_dataset
from transform import truncate_outliers_inplace, create_date_inplace, cols_to_scale, standardize

training = pd.read_csv("data/training_xgboost.csv")
test = pd.read_csv("data/test_ver2.csv")

Y = training[["fecha_dato", "ncodpers"] + y_cols]
training = training.drop(y_cols, axis=1)

training = training.append(test, ignore_index=True)

training = cleanup_dataset(training)

training.to_pickle("data/training_xgboost_clean.pickle")
Y.to_pickle("data/target_xgboost.pickle")


X_train_june = training[training["fecha_dato"] == "2015-06-28"]
y_train = Y[Y["fecha_dato"] == "2015-05-28"].drop("fecha_dato", axis=1)

X_train_june = X_train_june.merge(y_train, how="left", on="ncodpers")
X_train_june.fillna(0, inplace=True)

X_train_june.to_pickle("data/x_train_xgboost.pickle")

y_train = Y[Y["fecha_dato"] == "2015-06-28"].drop("fecha_dato", axis=1)
deltas = calculate_delta(X_train_june[y_cols].values, y_train[y_cols].values)
deltas = pd.DataFrame(deltas, columns=y_cols)

deltas.to_pickle("data/y_train_xgboost.pickle")
