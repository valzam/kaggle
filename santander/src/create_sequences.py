import pandas as pd
import pickle

y_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
          'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
          'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
          'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
          'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
          'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
          'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
          'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

def create_sequence(df):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    admin_cols = ['fecha_dato', 'ncodpers']
    unique_ids = df["ncodpers"].unique()

    for idx, user in enumerate(unique_ids):
        user_obs = df[df["ncodpers"] == user]
        X = user_obs.drop(admin_cols+y_cols, axis=1).values
        y = user_obs[y_cols].values
        X_train.append(X[:-1])
        y_train.append(y[:-1])
        X_test.append(X[-1:])
        y_test.append(y[-1:])

        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        if idx%100 == 0:
            print(idx)

    return X_train, y_train , X_test, y_test


if __name__ == "__main__":
    df = pd.read_pickle("../data/training_transformed.pickle")

    X_train, y_train , X_test, y_test = create_sequence(df)
    with open("../data/X_train.pickle", "wb") as handler:
        pickle.dump(X_train, handler)

    with open("../data/X_test.pickle", "wb") as handler:
        pickle.dump(X_test, handler)

    with open("../data/y_train.pickle", "wb") as handler:
        pickle.dump(y_train, handler)

    with open("../data/y_test.pickle", "wb") as handler:
        pickle.dump(y_test, handler)

    print(len(X_train))
    print(len(y_train))