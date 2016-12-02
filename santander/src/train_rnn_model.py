from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from datetime import datetime
import numpy as np
import pickle
from keras.preprocessing import sequence

from average_precision import calculate_delta

X_train = pickle.load( open( "data/X_train.pickle", "rb" ) )
y_train = np.array(pickle.load( open( "data/y_train.pickle", "rb" ) ))

X_train =sequence.pad_sequences(X_train, maxlen=5, dtype='float')

y_train = np.array(calculate_delta(X_train, y_train))
print(y_train[0:100])

# CONSTRUCT MODEL
model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

model.fit(X_train, y_train, nb_epoch=10, batch_size=512)

fp = "models/lstm_{0}_FULL".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
model.save(fp)