#deep_learning with python, tensorflow and keras

import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
from tensorboard import program

tf.disable_v2_behaviour()


NAME = "Cats-vs-dogs-cnn-64x2-{}".format(int(time.time()))
tracking_address = 'logs/{}'.format(NAME) # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


#tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

gpu_options= tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = np.array(X)
y=np.array(y)

#to normalize the data from classification_keras.py
X = X/255.0

model =Sequential()
#layer1
model.add(Conv2D(64,(3,3), input_shape = X.shape[1:]) )#convolution
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2))) #pooling

#layer2
model.add(Conv2D(64,(3,3), input_shape = X.shape[1:])) #convolution
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#layer3
model.add(Flatten()) #this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))

#output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,y, batch_size=32, epochs=10, validation_split=0.3, callbacks= [tb])