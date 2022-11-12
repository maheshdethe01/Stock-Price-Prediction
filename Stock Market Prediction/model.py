# Importing the libraries
import numpy as np
import pandas as pd
import pickle 
import datetime
from flask import Flask,request, render_template
import tensorflow
path = r"C:\Users\MAHESH DETHE\Downloads\MW-NIFTY-50-04-Nov-2022.csv"
nifty = pd.read_csv(path)
comp_data = nifty['SYMBOL \n'][1:].values.tolist()    #  list of Nifty 50 companies
main = pd.read_csv(r"C:\Users\MAHESH DETHE\Downloads\stock_price_data")
data = main[main['Name'] == 'TCS']             # data for training the model
opn = data[['Open']]
df = opn.values
from sklearn.preprocessing import StandardScaler
# Using MinMaxScaler for normalizing data 
mm = StandardScaler()
scaled_df = mm.fit_transform(np.array(df).reshape(-1,1))  #df is list of open stock prices of the company
#Defining test and train data sizes
train_size = int(len(scaled_df)*0.70)
test_size = len(scaled_df) - train_size    
#Splitting data between train and test
ds_train, ds_test = scaled_df[0:train_size,:], scaled_df[train_size:len(scaled_df),:1]     
# we are splitting it without sklearn cause it will take random dates but we want sequential records .
# Creating dataset in time series for LSTM model 
def create_df(dataset,k):
    Xtrain, Ytrain = [] , []
    for i in range(len(dataset)-k-1):
        new_df = dataset[i:(i+k), 0]
        Xtrain.append(new_df)
        Ytrain.append(dataset[i + k, 0])
    return np.array(Xtrain), np.array(Ytrain)
# Taking 100 days price as one record for training
time_stamp = 100
X_train, y_train = create_df(ds_train,time_stamp)
X_test, y_test = create_df(ds_test,time_stamp)
# Reshaping data to fit into LSTM model
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
# importing LSTM 
from keras.models import Sequential
from keras.layers import Dense, LSTM
# Creating LSTM model using keras
lstm = Sequential()
lstm.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
lstm.add(LSTM(units=50,return_sequences=True))
lstm.add(LSTM(units=50))
lstm.add(Dense(units=1,activation='linear'))
lstm.summary()
#Training model with adam optimizer and mean squared error loss function
lstm.compile( loss = 'mean_squared_error', optimizer = 'adam' )
lstm.fit(X_train,y_train,validation_data = (X_test , y_test), epochs = 100, batch_size = 64 )


pickle.dump(lstm,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))