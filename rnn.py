import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# select open stock price column 
training_set = dataset_train.iloc[:,1:2].values

'''
Feature Scaling
Normalisation
'''

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)


'''
Data Structure of 60 Time Steps and 1 Output
'''

x_train=[]
y_train=[]

for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

#convert DS to numpy array
x_train , y_train = np.array(x_train) , np.array(y_train)

# Reshaping x_train (input of NN) as Recurrent Layer takes 3D INput
'''
Add more indicator in future , currently using only 1
i.e open stock price 
 
'''
# Reshaping into 3D - Batch Size , time steps, input dim (i.e 1 [open stock price])
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


#importing Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising RNN
regressor = Sequential()

# Stacked LSTM 


regressor.add(LSTM(units=60, return_sequences=True , input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=40, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=40))
regressor.add(Dropout(0.2))

# Adding output layer
regressor.add(Dense(1))

#Compiling
regressor.compile( optimizer='adam', loss='mean_squared_error' )

#Fitting the RNN
regressor.fit(x_train,y_train,epochs=110,batch_size=32)

# saving
regressor.save('stock_redictor.h5')
print('model saved HDF5')



'''
Making the predictions
'''
# Getting real stock price of jan
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017

dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []

for i in range(60,80):
    x_test.append(inputs[i-60:i,0])

x_test =np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()






