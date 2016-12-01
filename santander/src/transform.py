import pandas as pd
from sklearn.preprocessing import MinMaxScaler

cols_to_scale = ["age", "renta", "antiguedad"]


def truncate_outliers_inplace(df):
    #cutoff = df["renta"].quantile(0.95) # 232843.5
    df.ix[df["renta"] > 232843, "renta"] = 232843.5

    #cutoff = df["antiguedad"].quantile(0.94) # 46
    df.ix[df["antiguedad"] > 46, "antiguedad"] = 46

    #cutoff = df["age"].quantile(0.94) # 71
    df.ix[df["age"] > 71, "age"] = 71

def standardize(df):

    df_scaled = df[cols_to_scale]
    df_scaled = MinMaxScaler().fit_transform(df_scaled)

    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = cols_to_scale

    return df_scaled

def create_date_inplace(df):
    df["fecha_dato"] = pd.to_datetime(df["fecha_dato"])

if __name__ == "__main__":
    df = pd.read_pickle("data/training.pickle")
    truncate_outliers_inplace(df)

    df_scaled = standardize(df)
    df.drop(cols_to_scale, axis=1, inplace=True)
    df = df.join(df_scaled)

    create_date_inplace(df)

    df.to_pickle("data/training_transformed.pickle")