

#import the required libraries
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


# In[57]:


#load the MNIST data and split it into train and test sets
(X_train,y_train), (X_test, y_test) = mnist.load_data()


# In[58]:


#get the image shape
print(X_train.shape)
print(X_test.shape)


# In[59]:


#print image shape
#X_train[0]


# In[60]:


#Print the image label
y_train[0]


# In[61]:


#plot the image via imshow
image_train = X_train[0]   
image_train = np.array(image_train, dtype='float')   
pixelst = image_train.reshape((28,28))  
plt.imshow(pixelst, cmap='gray')   
plt.show()


# In[62]:


#reshape the data to fit the model
X_train = X_train.reshape(60000, 28,28,1)
X_test = X_test.reshape(10000, 28, 28, 1)


# In[63]:


#one-Hot Encoding for fitting it for the model
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#Print the new label
print(y_train_one_hot[0])


# In[ ]:


model = Sequential()
#Convolutional layer
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
#max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#dense layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
#Compile the model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
hist = model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=5)


# In[65]:


#Visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy'], loc='upper left')
plt.show()


# In[66]:


#Visualize the models loss
plt.plot(hist.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper right')
plt.show()


# In[67]:


#predict the data
predictions = model.predict(X_test[:4])
predictions


# In[68]:



#Print the actual labels
print("Actual: ",y_test[:4])
#Print our predicitons as number labels for the first 4 images
print("Prediction: ", np.argmax(predictions, axis=1))


# In[69]:


#show the first 4 images as a pictures 
for i in range(0,4):   
    image = X_test[i]   
    image = np.array(image, dtype='float')   
    pixels = image.reshape((28,28))  
    plt.imshow(pixels, cmap='gray')   
    plt.show()

