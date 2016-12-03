import pandas as pd
import pickle

from create_sequences import create_sequence
from preprocess import cleanup_dataset
from transform import truncate_outliers_inplace, create_date_inplace, cols_to_scale, standardize

SUFFIX = "2016"

raw_input_csv_file = "data/training_2016.csv"

df = pd.read_csv(raw_input_csv_file)
df = cleanup_dataset(df)

df.to_pickle("data/training.pickle")

truncate_outliers_inplace(df)
df_scaled = standardize(df)
df.drop(cols_to_scale, axis=1, inplace=True)
df = df.join(df_scaled)
create_date_inplace(df)

df.to_pickle("data/training_transformed.pickle")

X_train, y_train = create_sequence(df)
with open("data/X_train.pickle", "wb") as handler:
    pickle.dump(X_train, handler)

with open("data/y_train.pickle", "wb") as handler:
    pickle.dump(y_train, handler)

print(len(X_train))
print(len(y_train))

