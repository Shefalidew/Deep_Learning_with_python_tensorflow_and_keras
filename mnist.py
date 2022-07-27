#deep_learning with python tensorflow and keras

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behaviour()

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test, y_test) = mnist.load_data()

#normalizing the data
x_train = tf.keras.utils.normalize(x_train, axis =1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#creating the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = 'adam', loss ='sparse_categorical_crossentropy',metrics=['accuracy'])

#to train the model
model.fit(x_train, y_train, epochs = 3)

#to validate or check overfit with the test dataset
val_loss, val_acc= model.evaluate(x_test,y_test)
print(val_loss,val_acc)

#to save the model
model.save('epic_num_reader.model')
#to load any model
new_model =tf.keras.models.load_model('epic_num_reader.model')
#to predict the data
predictions = new_model.predict([x_test])
print(predictions)
print(np.argmax(predictions[1]))

#print(x_train[0].shape)
plt.imshow(x_test[1], cmap=plt.cm.binary)
plt.show()