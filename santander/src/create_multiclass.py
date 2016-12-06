import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

def create_multiclass_dataset(X_train, y_train):
    print("Creating multiclass dataset")
    y_train["ncodpers"] = X_train["ncodpers"]
    y_train = y_train.set_index("ncodpers")

    # y_train is a dataframe with 24 columns
    # Each column value indicates whether the product was added in 2015-06,
    # in comparison to 2015-05

    # Create a multi-index, based on ncodpers and each columns
    # Reindex to create a "long" dataframe (one row for each (product, ncodpers) tuple
    stacked = y_train.stack()
    filtered_y_train = stacked.reset_index()
    filtered_y_train.columns = ["ncodpers", "product", "newly_added"]

    # Only take the rows where a product was added
    filtered_y_train = filtered_y_train[filtered_y_train["newly_added"] == 1]

    # Merge with the training set
    multiclass_training = filtered_y_train.merge(X_train, on="ncodpers", how="left")

    # Create categories as integers
    le = LabelEncoder()
    le.fit(multiclass_training["product"])
    with open("models/label_encoder.pickle", "wb") as handler:
        pickle.dump(le, handler)

    multiclass_training["target"] = le.transform(multiclass_training["product"])
    multiclass_training.drop("newly_added", inplace=True, axis=1)

    return multiclass_training


if __name__ == "__main__":
    X_train = pd.read_pickle("data/x_train_xgboost.pickle")
    y_train = pd.read_pickle("data/y_train_xgboost.pickle")

    full_train = create_multiclass_dataset(X_train, y_train)

    full_train.to_pickle("data/multiclass_training.pickle")