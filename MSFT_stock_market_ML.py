# The majorirty of code was taken from https://www.geeksforgeeks.org/stock-price-prediction-project-using-tensorflow/, writen by jaintarun, but parts have been added and modified

# What I learned:
# As this was my first time diving into machine learning, I learned a lot about how machine learning and neural networks work. While I still do not know
# a lot about machine learning, I learned a lot of the overview and gained an insight into what the code looks like and how it is supposed to opperate.
# I used Tensorflow in order to accomplish the machine learning. I was able to model to predict closing price for Microsoft's stock after the training data and 
# I could compare it to the testing data. However, I was unable to actually get the model to predict future data.

# I learned how to use major Python libraries, such as NumPy, Matplotlib, and Pandas. I used NumPy for arrays and reshaping arrays. I learned
# how to plot data using Matplotlib. The plotting took a while to learn, especially having multiple plots on one screen. However, once I learned the documentation,
# it was pretty simple to make plots after that. I also leanred how to use and manipulate a csv file using Pandas using Dataframes.


# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Getting data
df = pd.read_csv("MSFT.csv")
df['Date'] = pd.to_datetime(df['Date']) # Converting dates to DateTime data type

# Helps us to understand the data better
print(f"df Shape: {df.shape}")
print(f"df Info: {df.info()}")


# Exploratory Data Analysis (EDA) 

print(df.isnull().sum()) # Checks to make sure that none of our data is 0 and therefore useless

# Plots all six measures of MSTF's data from 3/13/1986 to 11/7/2023
cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
fig, axs = plt.subplots(2, 3) # Divides the plots in the window into a 2x3 arrangment to make it easier to see each graph
# Funtion plots each data measure
def axs_plot(row, col, index, color):
    axs[row,col].plot(df['Date'], df[cols[index]], f'tab:{color}', label=cols[index])
    axs[row,col].set_title(cols[index])
axs_plot(0, 0, 0, 'green')
axs_plot(0, 1, 1, 'blue')
axs_plot(0, 2, 2, 'brown')
axs_plot(1, 0, 3, 'red')
axs_plot(1, 1, 4, 'orange')
axs_plot(1, 2, 5, 'purple')

fig.suptitle('MSFT Stock')
plt.show()


# Removes columns from the data as we will not use them in our neural network 
df = df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'])

# As we are going to try and forecast future 'Close' data, we select only the 'Close' data
close_data = df.filter(['Close'])
df_close = close_data.values

training_length = int(np.ceil(len(df_close) * .80)) # Selects the training data as the first 95% of data

# Scales the close data from values 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1)) # Used MinMaxScaler (vs StandardScaler) as vaules far away from the mean and well known 
scaled_data = scaler.fit_transform(df_close)

train_data = scaled_data[0:training_length, :] # Puts the scaled close data into a list, ", :" because each element is held in single element list 
# Prepare feature(inputs) and labels(outputs)
x_train = []
y_train = []
 
#  Adds 60 elements of the training data to the features list and the next one to the label list 
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train) # Casts the feature and label lists to NumPy arrays
# Removes the third dimension of the x_train NumPy array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  


# Creates the model machine learning model using Tensorflow
# It uses Long Short Term Memory (LSTM) recurrent neural network as LSTM are better compared to other types of
#   neural networks as it does does better with information over a longer period of time
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
# model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=10)

# Gathers the test data
test_data = scaled_data[training_length - 60:, :]
x_test = []
y_test = df_close[training_length:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
 
#  Formats the test data that the model will predict against
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# predict the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) # Scales the data from values between 0 and 1 to the actual stock prices


# Evaluation metrics
mse = np.mean(((predictions - y_test) ** 2))
print("MSE", mse)               # "Mean Squared Error" - Measures the mean error between the prediction and real data
print("RMSE", np.sqrt(mse))     # "Rooted MSE" - another way to find error

# Gathers the parts of the data to plot
train = df.copy()
test = df.copy()
train.drop(train.index[training_length: 9491], inplace=True)
test.drop(test.index[0: training_length], inplace=True)
test['Predictions'] = predictions

# Plots the orginal data and the new predicted data
plt.figure(figsize=(10, 8))
plt.plot(train['Date'], train['Close'])
plt.plot(test['Date'], test[['Close', 'Predictions']])
plt.title('Microsoft Stock Close Price')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()

# Creates a plot that is zoomed in to better see the predictions
plt.figure(figsize=(10, 8))
plt.plot(train['Date'].iloc[8522:training_length], train['Close'].iloc[8522:training_length])
plt.plot(test['Date'], test[['Close', 'Predictions']])
plt.title('Microsoft Stock Close Price')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()
