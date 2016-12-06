import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

def create_multiclass_dataset(X_train, y_train):
    y_train["ncodpers"] = X_train["ncodpers"]
    y_train = y_train.set_index("ncodpers")

    stacked = y_train.stack()
    filtered_y_train = stacked.reset_index()
    filtered_y_train.columns = ["ncodpers", "product", "newly_added"]

    filtered_y_train = filtered_y_train[filtered_y_train["newly_added"] == 1]
    multiclass_training = filtered_y_train.merge(X_train, on="ncodpers", how="left")

    # Create categories as integers
    le = LabelEncoder()
    le.fit(multiclass_training["product"])
    with open("models/label_encoder.pickle", "wb") as handler:
        pickle.dump(le, handler)

    multiclass_training["target"] = le.transform(multiclass_training["product"])

    return multiclass_training


if __name__ == "__main__":
    X_train = pd.read_pickle("data/x_train_xgboost.pickle")
    y_train = pd.read_pickle("data/y_train_xgboost.pickle")

    full_train = create_multiclass_dataset(X_train, y_train)

    full_train.to_pickle("data/multiclass_training.pickle")