import os
import pandas as pd
import numpy as np
import cv2
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler


training_data=list()
target_data=list()

folder_path=os.path.join(os.getcwd(),'UCMerced_LandUse','Images')
target=dict()
count=1
for i in os.listdir(folder_path):
    target[i]=count
    count=count+1

for directory in os.listdir(folder_path):
    temp_folder=os.path.join(folder_path,directory)
    list_files=os.listdir(temp_folder)
    for file_name in list_files:
        try:
            image_path=os.path.join(temp_folder,file_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dsize = (100, 100)
            output = cv2.resize(gray, dsize)
            training_data.append(output.flatten())
            target_data.append(target[directory])
            print("Preprocessing image ",file_name)
        
        except:
            continue
    
    

#training_data=pd.DataFrame(training_data)
#target_data=pd.DataFrame(target_data)
#training_data.to_csv(os.path.join("data","training_data.csv"))
#target_data.to_csv(os.path.join("data","target_data.csv"))

X_train, X_test, y_train, y_test = train_test_split(np.array(training_data),np.array(target_data), random_state=0, test_size = 0.2 )

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X_train)
#scalarY.fit(y_train)
X_train = scalarX.transform(X_train)
#y_train = scalarY.transform(y_train)

scalarX.fit(X_test)
#scalarY.fir(y_test)
X_test = scalarX.transform(X_test)
#y_test = scalarY.transform(y_test)

batch_size = 10
epochs = 10
# input image dimensions
img_rows, img_cols = 100, 100
#inputshape = X.shape[1]

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score)

X=[]
y=[]

for i in range(1,2002):
    image_path=os.path.join(os.getcwd(),'wiki_judge_images','wiki_judge_images',str(i)+'.png')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dsize = (78, 78)
    output = cv2.resize(gray, dsize)
    X.append(output.flatten())

X=np.array(X)
X = X.astype('float32')
scalarX.fit(X)
X = scalarX.transform(X)
X = X.reshape(X.shape[0], img_rows, img_cols, 1)


y_pred=list(model.predict(X))

with open("result.csv","a+",newline='') as file_name:
    csv_writer=csv.writer(file_name)
    csv_writer.writerow(['ID','Age'])
    for i in range(0,len(y_pred)):
        csv_writer.writerow([int(i+1),float(y_pred[i])])
        