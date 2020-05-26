
# # LSTM Bitcoin stock price prediction

# In[1]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from keras import optimizers
from sklearn.model_selection import train_test_split


# ## Load the data

# In[2]:


df = pd.read_csv("C:/Users/Harry/Desktop/Analytics/BTC stock price.csv")

df["Open_before"] = df["Open"].shift(-1)
df["Open_changes"] = (df["Open"] / df["Open_before"]) - 1

df["High_before"] = df["High"].shift(-1)
df["High_changes"] = (df["High"] / df["High_before"]) - 1

df = df.dropna()

changes = df[["Open_changes", "High_changes"]].values
df.head()


# In[3]:


X = []
y = []

for i in range(0, len(changes) - 20):
    y.append(changes[i,0])
    X.append(np.array(changes[i+1:i+21][::-1]))


X = np.array(X).reshape(-1, 20, 2)
y = np.array(y)


# In[4]:


# Split the data up in train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## Training the model 

# In[17]:


model = Sequential()
#LSTM layer
model.add(LSTM(1, input_shape=(20, 2)))
# Adding the output layer
model.add(Dense(units = 1))
model.compile(optimizer="adam", loss="mse",metrics=['accuracy','mae'])
model.fit(X_train, y_train, batch_size=32, epochs=10)


# In[6]:


model.summary()


# ## Evaluating the model

# In[7]:


#evaluate the model
score = model.evaluate(X_test, y_test)
print(score)


# In[8]:


predictions = model.predict(X_test)


# In[9]:


predictions = predictions.reshape(-1)


# In[10]:


predictions = np.append(predictions, np.zeros(20))


# In[11]:


df["predictions"] = predictions


# In[12]:


df.head()


# In[13]:


df["Open_predicted"] = df["Open_before"] * (1 + df["predictions"])


# ## Ploting the result

# In[14]:



plt.figure(figsize=(30,10))
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

# format the ticks
ax = plt.gca()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

dates = np.array(df["Date"]).astype(np.datetime64)

plt.plot(dates, df["Open"], label="Open")
plt.plot(dates, df["Open_predicted"], label="Open (predicted)")


plt.title(' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
plt.show()


# In[15]:


#take a look into the last 100 days

plt.figure(figsize=(30,10))
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

# format the ticks
ax = plt.gca()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

dates = np.array(df["Date"]).astype(np.datetime64)

plt.plot(dates[-100:], df["Open"][-100:], label="Open")
plt.plot(dates[-100:], df["Open_predicted"][-100:], label="Open (predicted)")



plt.title(' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

