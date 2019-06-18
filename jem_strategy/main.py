import numpy as np
import decimal
import random
from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *
from QuantConnect.Data.Market import TradeBar
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from numpy.random import seed
from datetime import date

class Forex_Trade(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2018,11,7)    #Set Start Date
        self.SetEndDate(2018,11,21)    #Set End Date
        self.SetCash(100000)           #Set Strategy Cash
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Cash)
        self.currency = "EURUSD"
        self.currency2 = "AUDUSD"
        self.currency3 = "GBPUSD"

        self.resolution = Resolution.Hour
        self.AddForex(self.currency,self.resolution)
        self.AddForex(self.currency2,self.resolution)
        self.AddForex(self.currency3,self.resolution)
        self.AddForex("EURUSD 8G",self.resolution)

        self.long_list = []
        self.short_list = []
        self.model =Sequential()
        
        self.timeframe = 72
        self.indicatorData=[]
        self.start = 0
        self.features=6

        self.quoteBarWindow = RollingWindow[QuoteBar](self.timeframe)
        
        self.macd = self.MACD(self.currency, 12, 26, 5, MovingAverageType.Exponential, self.resolution)
        self.macd.Updated += self.MacdUpdated
        self.macdWin = RollingWindow[IndicatorDataPoint](self.timeframe)
        
        self.rsi = self.RSI(self.currency, 5, self.resolution)
        self.rsi.Updated += self.RsiUpdated
        self.rsiWin = RollingWindow[IndicatorDataPoint](self.timeframe)

        self.bob = self.BB(self.currency, 5, self.resolution)
        self.bob.Updated += self.BOBUpdated
        self.bobWin = RollingWindow[IndicatorDataPoint](self.timeframe)
        
        self.roc = self.ROC(self.currency, 5, self.resolution)
        self.roc.Updated += self.ROCUpdated
        self.rocWin = RollingWindow[IndicatorDataPoint](self.timeframe)
        
        self.IndicatorScaler = MinMaxScaler()
        
    def MacdUpdated(self, sender, updated):
        self.macdWin.Add(updated)
    
    def RsiUpdated(self, sender, updated):
        self.rsiWin.Add(updated)
        
    def BOBUpdated(self, sender, updated):
        self.bobWin.Add(updated)
    
    def ROCUpdated(self, sender, updated):
        self.rocWin.Add(updated)
        
    def OnData(self, data): # Minute
        originaldata=data
        self.quoteBarWindow.Add(data[self.currency])
        # Wait for windows to be ready.
        if not (self.quoteBarWindow.IsReady):
            return
        if not self.macdWin.IsReady:
            self.Debug("MACD not ready.")
            return
        if not self.rsiWin.IsReady:
            self.Debug("RSI not ready.")
            return
        if not self.bobWin.IsReady:
            self.Debug("BOB not ready.")
            return
        if not self.rocWin.IsReady:
            self.Debug("ROC not ready.")
            return
        
        if self.start == 0:
            #Create X training set 
            currency2_data = self.History([self.currency2], self.timeframe+1, self.resolution) # Asking for last 600 minutes of data
            data2 = np.array([currency2_data.close])
            currency2_data = data2[:,1:]
            
            currency3_data = self.History([self.currency3], self.timeframe+1, self.resolution) # Asking for last 600 minutes of data
            data3 = np.array([currency3_data.close])
            currency3_data = data3[:,1:]
            #self.Debug("currency2_data[0]: "+ str(currency2_data[0]))
            #self.Debug("currency2_data[0][0]: "+ str(currency2_data[0][0]))
            
            for i in range(self.timeframe):
                tempArray = []
                
                currMacd = self.macdWin[i]
                tempArray.append(currMacd.Value)
                #self.Debug("MACD:   {0} -> {1}".format(currMacd.Time, currMacd.Value))
                
                currRsi = self.rsiWin[i]
                tempArray.append(currRsi.Value)
                #self.Debug("RSI:   {0} -> {1}".format(currRsi.Time, currRsi.Value))

                currBob = self.bobWin[i]
                tempArray.append(currBob.Value)
                #self.Debug("SMA:   {0} -> {1}".format(currBob.Time, currBob.Value))
                
                currRoc = self.rocWin[i]
                tempArray.append(currRoc.Value)
                #self.Debug("SMA:   {0} -> {1}".format(currRoc.Time, currRoc.Value))
                
                tempArray.append(currency2_data[0][i])
                tempArray.append(currency3_data[0][i])

                self.indicatorData.append(tempArray)
                
            self.Debug(len(currency2_data))

            #Scale and normalise x training data    
            #self.Debug("Length of training data: "+str(len(self.indicatorData)))
            #self.Debug("X_data is " + str(self.indicatorData))
            self.IndicatorScaler.fit(self.indicatorData)
            X_data = self.IndicatorScaler.transform(self.indicatorData)
            #self.Debug("________________________________________________________")
            #self.Debug("X_data after transform: "+str(X_data))
            
            X_corr = pd.DataFrame(X_data)
            #self.Debug("X_Corr :"+str(X_corr))
            corr_matrix = X_corr.corr()
            self.Debug(corr_matrix)
            
            #Create Y training set
            currency_data = self.History([self.currency], self.timeframe+1, self.resolution) # Asking for last 600 minutes of data
            L = len(currency_data)
            #self.Debug("currency data is " + str(currency_data))
            if not currency_data.empty:
                data = np.array([currency_data.close])  #Get the close prices and make an array
                #self.Debug("Y data is " + str(data))
                    
                #Calculate price direction up or down
                prev_data = data[:,0:-1]
                curr_data = data[:,1:]
                #self.Debug(curr_data - prev_data)
                Y_data = np.transpose(curr_data - prev_data)
                    
                #self.Debug("Y after transpose is " + str(Y_data))
                #scaler1 = MinMaxScaler()
                #scaler1.fit(Y_data)
                #Y_data = scaler1.transform(Y_data)
                #self.Debug("________________________________________________________")
                #self.Debug("Y_data: " + str(Y_data))
                #self.Debug("X data length is " + str(len(X_data)))
                #self.Debug("Y data length is " + str(len(Y_data)))
            
        
        if self.start == 0:
            
            X_data_final = X_data
            Y_data_final = Y_data
            
            #Oversampling of the last j data points, proportionally
            #The last data point will be oversampled j times
            #The second last data point will be oversampled j-1 times... and so on
            j=10
            for i in range (j):
                X_data_final= np.vstack((X_data_final, np.repeat([X_data[self.timeframe-j+i]],repeats= i ,axis=0)))
                Y_data_final= np.vstack((Y_data_final, np.repeat([Y_data[self.timeframe-j+i]],repeats= i ,axis=0)))
                    
            self.Debug("X data final length is " + str(len(X_data_final)))
            self.Debug("Y data final length is " + str(len(Y_data_final)))
            #self.Debug("X data final " + str(X_data_final))
            #self.Debug("Y data final " + str(Y_data_final))

            #USE TimeSeriesSplit to split data into n sequential splits
            tscv = TimeSeriesSplit(n_splits=2)
                
            # Make cells and epochs to be used in grid search.
            cells = [100,200]
            epochs  = [100,200]
                
            # creating a datframe to store final results of cross validation for different combination of cells and epochs
            df = pd.DataFrame(columns= ['cells','epoch','mse'])
                
            #Loop for every combination of cells and epochs. In this setup, 4 combinations of cells and epochs [100, 100] [ 100,200] [200,100] [200,200]
            for i in cells:
                for j in epochs:
                    cvscores = []
                    # to store CV results
                    #Run the LSTM in loop for every combination of cells an epochs and every train/test split in order to get average mse for each combination.
                    for train_index, test_index in tscv.split(X_data):
                        #self.Debug("TRAIN:", train_index, "TEST:", test_index)
                        X_train, X_test = X_data[train_index], X_data[test_index]
                        Y_train, Y_test = Y_data[train_index], Y_data[test_index]
                        #self.Debug ( " X train [0] is " + str (X_train[0]))
                        #self.Debug ( " X train [1] is " + str (X_train[1]))
                        
                        X_train= np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))                            
                        #self.Debug("X input to LSTM :  " + str(X_train))
                        X_test= np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
                        #self.Debug("Y input to LSTM :  "+ str(Y_train))
                
                        #self.Debug("START: LSTM Model")
                        #self.Debug(i)
                        #self.Debug(j)
                        model = Sequential()
                        model.add(LSTM(i, input_shape = (1,self.features), return_sequences = True))
                        model.add(Dropout(0.10))
                        model.add(LSTM(i,return_sequences = True))
                        model.add(LSTM(i))                            
                        model.add(Dropout(0.10))
                        model.add(Dense(1))
                        model.compile(loss= 'mean_squared_error',optimizer = 'rmsprop', metrics = ['mean_squared_error'])
                        model.fit(X_train,Y_train,epochs=j,verbose=0)
                        #self.Debug("END: LSTM Model")
                        
                        scores = model.evaluate(X_test, Y_test, verbose=0)
                        #self.Debug("%s: %f " % (model.metrics_names[1], scores[1]))
                        cvscores.append(scores[1])
                                
                    MSE= np.mean(cvscores)
                    #self.Debug("MSE" + str(MSE))
                    
                    #Create a dataframe to store output from each combination and append to final results dataframe df.
                    df1 = pd.DataFrame({ 'cells': [i], 'epoch': [j], 'mse': [MSE]})
                    #self.Debug("Individual run ouput DF1" + str(df1))
                    #Appending individual ouputs to final dataframe for comparison
                    df = df.append(df1) 
                        
            #self.Debug("Final table of DF"+ str(df))
                
            #Check the optimised values obtained from cross validation
            #This code gives the row which has minimum mse and store the values to O_values
            O_values = df[df['mse']==df['mse'].min()]
    
            # Extract the optimised  values of cells and epochs from above row (having min mse )
            O_cells = O_values.iloc[0][0]                
            O_epochs = O_values.iloc[0][1]
                
            self.Debug( "O_cells"  + str (O_cells))
            self.Debug( "O_epochs" + str (O_epochs))
                
            X_data1= np.reshape(X_data, (X_data.shape[0],1,X_data.shape[1]))                
            #self.Debug("START: Final_LSTM Model")
            
            self.model.add(LSTM(O_cells, input_shape = (1,self.features), return_sequences = True))
            self.model.add(Dropout(0.10))
            self.model.add(LSTM(O_cells,return_sequences = True))
            self.model.add(LSTM(O_cells))
            self.model.add(Dropout(0.10))
            self.model.add(Dense(1))
            self.model.compile(loss= 'mean_squared_error',optimizer = 'rmsprop', metrics = ['mean_squared_error'])
            self.model.fit(X_data1,Y_data,epochs=O_epochs,verbose=0)
            #self.Debug("END: Final_LSTM Model")
                
        self.start = 1
        
        #Build model for whole data:
        #Repeating the model but for optimised cells and epochs
        
        #self.Debug("currency 2 data:" +str(originaldata[self.currency2].Close))
        #Prepare new data for prediction based above model
        X_new = [[ self.macd.Current.Value,self.rsi.Current.Value,
                    self.bob.Current.Value, self.roc.Current.Value,
                    originaldata[self.currency2].Close, originaldata[self.currency3].Close]]
        #self.Debug("X_new data is "+str(X_new))
        X_new=self.IndicatorScaler.transform(X_new)
        #self.Debug("X_new data after transform is "+str(X_new))
        #self.Debug("Length of X_new: "+str(len(X_new)))
        
        X_new = np.reshape(X_new,(X_new.shape[0],1,X_new.shape[1]))
        #self.Debug(X_new)
            
        #Predicting with the LSTM model
        output = self.model.predict(X_new)
                
        #Need to inverse transform output 
        #self.Debug("Output from LSTM model is" + str(output))
        
        #Checking the current price 
        currency_data = self.History([self.currency], self.timeframe + 1, self.resolution)
        price = currency_data.close[-1]
        #self.Debug("Current price is" + str(price))
        

        ####Make decision for trading based on the output from LSTM and the current price.
        #Long when output exceeds a positive limit
        #Close long position when output becomes negative
        #Short when output falls below a negative limit
        #Close short position when output becomes positive
        
        upper_limit = 0.0000025
        lower_limit = upper_limit*-1
        upper_bound = 0.00025
        lower_bound = upper_bound*-1
        size = 0
        
        #Model predicts that price will rise by a lot. Full long
        if output > upper_bound:
            size = 1
        
        #Model predicts that price will rise, but not by a lot. Partial long
        elif output>upper_limit and output <upper_bound:
            size = min(1,output/upper_bound * 100)
        
        #Model predicts price will drop by a lot. Full short
        elif output < lower_bound: 
            size = 1
        
        #Model predicts price will drop, but not by a lot. Partial short
        elif output < lower_limit and output > lower_bound:
            #Go partial short
            size = min(1,output/lower_bound * 100)
            
        #Close long position if output becomes negative and holding long position
        if self.currency in self.long_list and output < 0:
            self.Liquidate(self.currency)
            #self.SetHoldings(self.currency, 0)
            self.long_list.remove(self.currency)
            self.Debug("Output from LSTM model is" + str(output))
            self.Debug("Close long position")
            
        if self.currency in self.short_list and output > 0:
            self.Liquidate(self.currency)
            self.short_list.remove(self.currency)
            self.Debug("Output from LSTM model is" + str(output))
            self.Debug("Close short position")
            
        #Long when output exceeds limit and not holding any position
        if output > upper_limit and self.currency not in self.long_list and self.currency not in self.short_list:
            self.SetHoldings(self.currency, size)
            self.long_list.append(self.currency)
            self.Debug("Output from LSTM model is" + str(output))
            self.Debug("Make a long position")
            
        #Short when output falls below limit and not holding any position
        if output < lower_limit and self.currency not in self.long_list and self.currency not in self.short_list:
            self.SetHoldings(self.currency, size*-1)
            self.short_list.append(self.currency)
            self.Debug("Output from LSTM model is" + str(output))
            self.Debug("Make a short position")
            
        #self.Debug("END: Ondata")