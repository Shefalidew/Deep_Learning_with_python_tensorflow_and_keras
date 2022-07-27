#deep_learning with python,tensorflow and keras


import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


DATADIR = "/home/student/Downloads/kagglecatsanddogs_5340/PetImages"
CATEGORIES = ["Dog","Cat"]

#for category in CATEGORIES:
    #path = os.path.join(DATADIR,category) #path to cat or dog dir
    #for img in os.listdir(path):
       # img_array = cv2.imread(os.path.join(path, img))
       # plt.imshow(img_array)
       # plt.show()
        #break
    #break

#print(img_array.shape)  #3dimesional array because of RGB

#to fix a size for the images as they might be of different sizes
IMG_SIZE=50
#new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
#plt.imshow(new_array)
#plt.show()

#to create a training dataset
training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to cat or dog dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()
print(len(training_data))

#to have a balanced datset we shuffle the datas
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X=[]
y=[]
for features, label in training_data:
    X.append(features)
    y.append(label)

X=np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

#to save the preprocessed data
pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()
pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

#pickle_in = open("X.pickle","rb")
#X=pickle.load(pickle_in)



