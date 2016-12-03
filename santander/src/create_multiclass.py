import pandas as pd

def create_multiclass_dataset(X_train, y_train):
    ids = X_train["ncodpers"]
    target_cols = y_train.head().columns
    y_train["ncodpers"] = ids

    new_x = []
    for count, cid in enumerate(list(ids)):
        if count%1000 == 0:
            print(count)

        x_row = X_train[X_train["ncodpers"] == cid].values[0]
        y_row = y_train[y_train["ncodpers"] == cid]
        for idx, c in enumerate(target_cols):
            val = y_row[c].values[0]
            if val == 1:
                print("Added row for customer id{0}".format(cid))
                new_x.append(list(x_row) + [idx])
    train = pd.DataFrame(new_x, columns=list(X_train.columns) + ["target"])

    return train


if __name__ == "__main__":
    X_train = pd.read_pickle("data/x_train_xgboost.pickle")
    y_train = pd.read_pickle("data/y_train_xgboost.pickle")

    full_train = create_multiclass_dataset(X_train, y_train)

    full_train.to_pickle("data/multiclass_training.pickle")