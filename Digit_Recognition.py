#import the required libraries
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


#loading the dataset and classifying in train and test sets
(trainX,trainY),(testX,testY)=mnist.load_data()



#visualizing the mnist dataset
plt.subplot(221)
plt.imshow(trainX[0],cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(trainX[1],cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(trainX[2],cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(trainX[3],cmap=plt.get_cmap('gray'))
plt.show()


#num_pixels=784
num_pixels=trainX.shape[1]*trainX.shape[2]
trainX = trainX.reshape((trainX.shape[0],num_pixels)).astype('float32')
testX = testX.reshape((testX.shape[0],num_pixels)).astype('float32')



#reshaping the data between 0 and1
trainX=trainX/255
testX=testX/255


#one hot encoding
trainY=np_utils.to_categorical(trainY)
testY=np_utils.to_categorical(testY)


#num_classes=784
num_classes=testY.shape[1]


#building and compiling the model through a function
def baseline_model():
    #creating model
    model=Sequential()
    model.add(Dense(num_pixels,input_dim=num_pixels,kernel_initializer='normal',activation='relu'))
    model.add(Dense(num_classes,kernel_initializer='normal',activation='softmax'))
    #compiling model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
    
model=baseline_model()   


#fitting the model
model.fit(trainX,trainY,validation_data=(testX,testY),epochs=10,batch_size=500,verbose=1)


#model evaluation
score=model.evaluate(testX,testY,verbose=0)
print('MODEL ACCURACY:',score[1]*100)
