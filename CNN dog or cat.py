#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dropout
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


# In[4]:


# Initialising the CNN
model = Sequential()


#  Convolution
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(Dropout(0.25))
# Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))
# Flattening
model.add(Flatten())

#Full connection
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/Harry/Desktop/Analytics/dataset cats and dogs/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:/Users/Harry/Desktop/Analytics/dataset cats and dogs/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#training the model
model.fit_generator(training_set,
                        
                         epochs = 25,
                         validation_data = test_set)
                   


# In[5]:


#evaluete the model
model.evaluate(test_set)


# ![image.png](attachment:image.png)
# 

# In[6]:


#Making new predictions by loading the test data


test_image = image.load_img('C:/Users/Harry/Desktop/Analytics/dataset cats and dogs/single_prediction/cat_or_dog_4.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    

# printing the actual picture and the prediction
print("Prediction: ",prediction) 
print("\nPicture: ") 

training_setnp = test_image.astype(np.float64) / 255.

plt.imshow(training_setnp[0])
plt.show()


# In[ ]:




