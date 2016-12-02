from keras.models import load_model
from src.average_precision import convert_to_names
from keras.preprocessing import sequence
from datetime import datetime
import pandas as pd
import numpy as np
import pickle

X = pickle.load( open( "data/training_full.pickle", "rb" ) )
ids = pickle.load( open( "data/training_ids.pickle", "rb" ) )
test_ids = pd.read_csv("data/test_ver2.csv", usecols=[1])

X =sequence.pad_sequences(X, maxlen=5, dtype='float')

model = "lstm_2016-12-02 23:38:41_FULL"
best_model_path = "models/{0}".format(model)
model = load_model(best_model_path)

preds = np.round(model.predict_proba(X, batch_size=512), 2)

predicted_names = []
for i, p in enumerate(preds):
    predicted_names.append(convert_to_names(p, 0.01))

submission =  pd.DataFrame()
submission["ncodpers"] = ids
submission["added_products"] = predicted_names

def flatten_list(arr):
    return " ".join(arr)

submission["added_products"] = submission["added_products"].apply(flatten_list)
final_submission = test_ids.merge(submission, how="left")
final_submission.to_csv("submissions/submission_{0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), index=False)