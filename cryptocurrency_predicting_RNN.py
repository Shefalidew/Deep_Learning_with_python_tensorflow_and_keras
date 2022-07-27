#deep_learning with p   ython tensorflow and keras
#very useful for my project to consider last 5 values of BB and predict 1 future trajectory for every second
import random

import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import  time
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LEN =60
FUTURE_PERIOD_PREDICT= 3
RATIO_TO_PREDICT ="LTC-USD"
EPOCHS =10
BATCH_SIZE= 64
NAME =f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop('future',1)

    for col in df.columns:
        if col != "target":
            df[col] =df[col].pct_change()
            df.dropna(inplace=True)
            df[col] =preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    sequential_data =[]
    prev_days = deque(maxlen =SEQ_LEN) # it creates a list, as this list reaches its max len its pops out the old value and appends new value
    #print(df.head())
    #for c in df.columns:
    #   print(c)
    for i in df.values:
        prev_days.append([n for n in i[:-1]]) # to exclude the last column i.e target
        if len(prev_days) ==SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    #print(len(sequential_data))
    random.shuffle(sequential_data)

    buys = []
    sells =[]
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        elif target==1:
            buys.append([seq,target])

    random.shuffle(buys)
    random.shuffle(sells)
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys +sells
    random.shuffle(sequential_data)

    X = []
    y=[]
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    return np.array(X), y

main_df = pd.DataFrame()
ratios = ["BTC-USD","LTC-USD","ETH-USD","BCH-USD"]
for ratio in ratios:
    dataset = f"/home/student/Downloads/crypto_data/{ratio}.csv"
    df = pd.read_csv(dataset, names =["time","low","high","open","close","volume"])
    #print(df.head())
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
    df.set_index("time", inplace =True)
    df = df[[f"{ratio}_close",f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df =df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] =list(map(classify,main_df[f"{RATIO_TO_PREDICT}_close"],main_df["future"]))
#print(main_df[[f"{RATIO_TO_PREDICT}_close","future", "target"]].head(20))

#to create the test dataset from the sample (special procedure for time series data)
times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]
#print(last_5pct)

validation_main_df = main_df[(main_df.index >= last_5pct)] #created testing data
main_df = main_df[(main_df.index < last_5pct)]

#preprocess_df(main_df)
train_x, train_y =preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

# to print all the statistics
print(f"train data: {len(train_x)}, validation: {len(validation_x)}")
print(f"Don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"Validation Don't buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

train_x =np.asarray(train_x)
train_y =np.asarray(train_y)
validation_x =np.asarray(validation_x)
validation_y =np.asarray(validation_y)

# to create model and train it
model =Sequential()

model.add(LSTM(128, input_shape =(train_x.shape[1:]), return_sequences =True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape =(train_x.shape[1:]), return_sequences =True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape =(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation= 'relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation= 'softmax'))

opt =tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

model.compile(loss= 'sparse_categorical_crossentropy', optimizer = opt, metrics= ['accuracy'])

history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS,validation_data= (validation_x, validation_y))


