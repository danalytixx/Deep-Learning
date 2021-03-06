

# # LSTM stock price prediction

# In[1]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import LSTM, Dense,Flatten, Dropout, Activation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler 
import matplotlib.cbook as cbook
from keras import optimizers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ## Load the data

# In[2]:


df = pd.read_csv("C:/Users/Harry/Desktop/Analytics/BTC stock price.csv")
df = df.dropna()


Open = df[["Open"]]
Open


changes = df[["Open", "High"]]
changes


# In[3]:


#Min-Max scaling - standardization (normalization)

scaler= MinMaxScaler(feature_range = (0, 1))

# To scale data 
stand=scaler.fit_transform(changes) 

df_stand=pd.DataFrame(stand, columns=["Open", "High"])

df_stand.shape


# In[4]:


#Min-Max scaling - standardization (normalization) only for Open price

scaler1= MinMaxScaler(feature_range = (0, 1))

# To scale data 
stand1=scaler1.fit_transform(Open) 
stand1.shape


# In[5]:


# using 20 days as a timeframe  
X = []
y = []

changes_num=df_stand.values
for i in range(0, len(changes_num) - 20):
    y.append(changes_num[i,0])
    X.append(np.array(changes_num[i+1:i+21][::-1]))
    
X = np.array(X).reshape(-1, 20, 2)
y = np.array(y)
X.shape


# In[6]:


# Split the data up in train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,shuffle=False)
X_test.shape


# ## Training the model 

# In[7]:


# Create NN
model = Sequential()
#LSTM layer
model.add(LSTM(64, activation='relu',return_sequences = True,input_shape = (20, 2)))
model.add(Dropout(0.01))
model.add(LSTM(32,activation='relu',return_sequences = False))
model.add(Dropout(0.01))
model.add(Dense(32))
model.add(Dense(16))

#output layer
model.add(Dense(1))

model.compile(optimizer="nadam", loss="mse",metrics=['accuracy'])
history=model.fit(X_train, y_train ,batch_size=16, epochs=100)


# ## Evaluating the model and predict

# In[8]:


#evaluate the model
score = model.evaluate(X_test, y_test)
print(score)


# In[9]:

#plot loss over number of epochs
plt.figure(figsize=(15,5))
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[10]:


#calculate the prediction
predictions = model.predict(X_test)
predictions.shape


# In[11]:


#scaling back to original data
predictions_Open=scaler1.inverse_transform(predictions)
predictions_Open=predictions_Open.reshape(-1)
predictions_Open=np.append(predictions_Open, np.zeros(20))
predictions_Open.shape


# ## Ploting the result

# In[12]:


plt.figure(figsize=(30,10))
dates = np.array(df["Date"]).astype(np.datetime64)

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

plt.plot(dates[:-len(predictions_Open)], df["Open"][:-len(predictions_Open)], label="Open Train")
plt.plot(dates[-len(predictions_Open):-20], df["Open"][-len(predictions_Open):-20], label="Open Test")
plt.plot(dates[-len(predictions_Open):-20], predictions_Open[-len(predictions_Open):-20], label="Open Test (predicted)")

plt.title(' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[13]:


plt.figure(figsize=(30,10))
dates = np.array(df["Date"]).astype(np.datetime64)


years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')


plt.plot(dates[-200:-20], df["Open"][-200:-20], label="Open Test")
plt.plot(dates[-200:-20], predictions_Open[-200:-20], label="Open Test (predicted)")

plt.title(' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



