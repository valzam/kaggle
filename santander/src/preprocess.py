import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as DV

cols_to_binarize = ["sexo", "indext", "indfall"]
cols_to_dummyfy = ["ind_empleado", "tiprel_1mes", "segmento"]
useless_cols_to_delete = ["pais_residencia", "indrel_1mes", "ult_fec_cli_1t", "canal_entrada", "fecha_alta", "tipodom",
                          "cod_prov", "nomprov"]
low_variance_cols_to_delete = ["conyuemp", "indresi", "indrel"]

def cleanup_dataset(df):
    print("Deleting missings and unused cols")
    fill_missings_inplace(df)
    delete_cols_inplace(df)
    print(df.dtypes)
    print("Binarizing")
    binary_features = binaryze(df)
    df.drop(cols_to_binarize, axis=1, inplace=True)
    df = df.join(binary_features)

    print("Dummyfying")
    dummy_features = dummyfy(df)
    df.drop(cols_to_dummyfy, axis=1, inplace=True)
    df = df.join(dummy_features)

    return df


def binaryze(df):
    binary_features = pd.DataFrame()

    for c in cols_to_binarize:
        lb = preprocessing.LabelBinarizer()
        binary_features[c] = lb.fit_transform(list(df[c])).flatten()

    return binary_features


def dummyfy(df):
    cat_df = df[cols_to_dummyfy]
    cat_dict = cat_df.T.to_dict().values()
    vectorizer = DV(sparse=False)
    dummy_features = pd.DataFrame(vectorizer.fit_transform(cat_dict))

    dummy_features.columns = vectorizer.feature_names_

    return dummy_features

def delete_cols_inplace(df):

    df.drop(useless_cols_to_delete,axis=1, inplace=True, errors="ignore")
    df.drop(low_variance_cols_to_delete,axis=1, inplace=True, errors="ignore")

def fill_missings_inplace(df):
    imputations = {
        "tiprel_1mes": "I",
        "conyuemp":0,
        "renta": 0,
        "segmento": "PARTICULARES",
        "ind_nom_pens_ult1": 0,
        "ind_recibo_ult1": 0,
        "ind_nomina_ult1": 0,
        "sexo": "H"

    }

    df.fillna(imputations, inplace=True)
    print("Missing values per column")
    print(pd.isnull(df).sum())

if __name__ == "__main__":
    df = pd.read_csv("../data/training_june_2015.csv")
    df = cleanup_dataset(df)
    df.to_pickle("../data/training_june_2015.pickle")