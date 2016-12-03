import pandas as pd

X = pd.read_pickle("data/training_xgboost_clean.pickle")
Y = pd.read_pickle("data/target_xgboost.pickle")

X_submission = X[X["fecha_dato"] == "2016-06-28"]
y_train = Y[Y["fecha_dato"] == "2016-05-28"].drop("fecha_dato", axis=1)

X_submission = X_submission.merge(y_train, how="left", on="ncodpers")
X_submission.fillna(0, inplace=True)
X_submission.drop(["fecha_dato"], axis=1, inplace=True)
X_submission.replace('         NA', 0, inplace=True)
X_submission["renta"] = X_submission["renta"].astype("float")

X_submission.to_pickle("data/x_submission_xgboost.pickle")