import pandas as pd
import xgboost
from datetime import datetime
from average_precision import convert_to_names

X_sub = pd.read_pickle("data/x_submission_xgboost.pickle")

model = xgboost.Booster({'nthread':4}) #init model
model.load_model("models/xgboost.bin") # load data

preds = model.predict(xgboost.DMatrix(X_sub.drop(["ncodpers"], axis=1).values))

test_ids = X_sub["ncodpers"]
predicted_names = []
for i, p in enumerate(preds):
    predicted_names.append(convert_to_names(p, 0.01))

final_submission =  pd.DataFrame()
final_submission["ncodpers"] = test_ids
final_submission["added_products"] = predicted_names

def flatten_list(arr):
    return " ".join(arr)

final_submission["added_products"] = final_submission["added_products"].apply(flatten_list)

final_submission.to_csv("submissions/submission_{0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), index=False)

print("The submission has {0} rows".format(len(final_submission)))
print("This is what the first few subs look like:")
print(final_submission.head())