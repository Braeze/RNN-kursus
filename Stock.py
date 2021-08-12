
# Part 1 - Data Preprocessing

#import libraies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras


#Gpu code
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
    
#Import train set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
train_set = dataset_train.iloc[:, 1:2].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
train_set_scaled = sc.fit_transform(train_set)

# Creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(train_set_scaled[i-60:i, 0])
    y_train.append(train_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Part 2 - Building the RNN
regressor = keras.models.Sequential()
#return_seq is because we add multiple layers
regressor.add(keras.layers.LSTM(units = 50 , return_sequences=True, input_shape = (x_train.shape[1], 1)))
regressor.add(keras.layers.Dropout(0.2))

regressor.add(keras.layers.LSTM(units = 50 , return_sequences=True))
regressor.add(keras.layers.Dropout(0.2))

regressor.add(keras.layers.LSTM(units = 50 , return_sequences=True))
regressor.add(keras.layers.Dropout(0.2))

regressor.add(keras.layers.LSTM(units = 50))
regressor.add(keras.layers.Dropout(0.2))

regressor.add(keras.layers.Dense(units = 1))

#compiling
regressor.compile(optimizer= 'adam', loss = 'mean_squared_error' )

#train it
regressor.fit(x_train, y_train, epochs = 200, batch_size = 32 )

# Part 3 - Making the predictions and visualising the results
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
test_set = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted = regressor.predict(x_test)
predicted = sc.inverse_transform(predicted)

# Visualising the results
plt.plot(test_set, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()