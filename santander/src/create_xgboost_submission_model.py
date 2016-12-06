import xgboost
import pandas as pd

X = pd.read_pickle("data/multiclass_training.pickle")

cols_not_for_training = ["fecha_dato","ncodpers", "target", "product"]
X_train = X.drop(cols_not_for_training, axis=1).values
y_train = X["target"].values

xgbtrain = xgboost.DMatrix(X_train, y_train)
del X_train, y_train

param = {}

param['objective'] = 'multi:softprob'
param['eta'] = 0.05
param['max_depth'] = 8
param['silent'] = 0
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.5
param['colsample_bytree'] = 0.5
param['colsample_bylevel'] = 0.5
param['seed'] = 323
num_rounds = 100

plst = list(param.items())
model = xgboost.train(plst, xgbtrain, num_rounds)

model.save_model("models/xgboost.bin")