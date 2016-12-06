import pandas as pd

from average_precision import calculate_delta
from create_multiclass import create_multiclass_dataset
from create_sequences import y_cols
from preprocess import cleanup_dataset

# Read raw datasets
training = pd.read_csv("data/training_xgboost.csv")
submission = pd.read_csv("data/test_ver2.csv")

# Create target vector and combine training and submission input
y = training[["fecha_dato", "ncodpers"] + y_cols]
training = training.drop(y_cols, axis=1)

X_full = training.append(submission, ignore_index=True)

# Basic transformations
X_full = cleanup_dataset(X_full)
X_full.to_pickle("data/training_xgboost_clean.pickle")
y.to_pickle("data/target_xgboost.pickle")

# Create X_train set
X_train = X_full[X_full["fecha_dato"] == "2015-06-28"]
previous_months_products = y[y["fecha_dato"] == "2015-05-28"]
X_train = X_train.merge(previous_months_products, how="left", on=["fecha_dato", "ncodpers"])
X_train.fillna(0, inplace=True)

X_train.to_pickle("data/x_train_xgboost.pickle")

# Create y_train deltas
y_train = y[y["fecha_dato"] == "2015-06-28"].drop("fecha_dato", axis=1)
deltas = calculate_delta(X_train[y_cols].values, y_train[y_cols].values)
deltas = pd.DataFrame(deltas, columns=y_cols)

deltas.to_pickle("data/y_train_xgboost.pickle")

# Create multiclass dataset for prediction
X_train_multiclass = create_multiclass_dataset(X_train, y_train)

X_train_multiclass.to_pickle("data/multiclass_training.pickle")
