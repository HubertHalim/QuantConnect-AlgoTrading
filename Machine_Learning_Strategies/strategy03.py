from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Indicators")
AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Data import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *
from datetime import timedelta
import tensorflow as tf
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics
from keras import backend as K

from datetime import datetime, timedelta

import pandas as pd

class AlgorithmOne(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the
        cash and start-end dates for your algorithm. All algorithms must initialized.'''

        # set cash
        self.SetCash(100000)

        # set dates
        self.SetStartDate(2018, 7, 6) # The first time US imposes China specific tariffs
        self.SetEndDate(2019, 6, 6)

        # set asset data (FOREX contracts in this case)
        self.currency = "EURUSD"
        self.resolution = Resolution.Daily # set to daily for testing as it takes too long to run minute
        self.AddForex(self.currency,self.resolution)

        # set indicators
        self.rsi = RelativeStrengthIndex(9)
        self.bb = BollingerBands(30,2,2)
        self.macd = MovingAverageConvergenceDivergence(12, 26, 9)
        self.stochastic = Stochastic(14, 3, 3)
        self.ema = ExponentialMovingAverage(9)
        
        ## define a long list, short list and portfolio
        self.long_l = []
        self.short_l = []
        
        prev_rsi = []
        prev_bb = []
        prev_macd = []
        low_bb = []
        up_bb = []
        sd_bb = []
        prev_stochastic = []
        prev_ema = []

        # Historical Currencty Data
        self.currency_data = self.History([self.currency,], 150, Resolution.Daily) # Drop the first 20 for indicators to warm up
        
        y_open = self.currency_data["open"][-1]
        y_close = self.currency_data["close"][-1]
        
        self.currency_data = self.currency_data[:-1]

        for tuple in self.currency_data.loc[self.currency].itertuples():        
            # making Ibasedatabar for stochastic
            bar = QuoteBar(
                tuple.Index, 
                "EURUSD",
                Bar(tuple.bidclose, tuple.bidhigh, tuple.bidlow, tuple.bidopen),0,
                Bar(tuple.askclose, tuple.askhigh, tuple.asklow, tuple.askopen), 0 ,timedelta(days=1)
            )
        
    
            self.stochastic.Update(bar)
            prev_stochastic.append(float(self.stochastic.ToString()))
    
            self.rsi.Update(tuple.Index, tuple.close)
            prev_rsi.append(float(self.rsi.ToString()))
    
            self.bb.Update(tuple.Index, tuple.close)
            prev_bb.append(float(self.bb.ToString()))
            low_bb.append(float(self.bb.LowerBand.ToString()))
            up_bb.append(float(self.bb.UpperBand.ToString()))
            sd_bb.append(float(self.bb.StandardDeviation.ToString()))
    
            self.macd.Update(tuple.Index, tuple.close)
            prev_macd.append(float(self.macd.ToString()))
            
            self.ema.Update(tuple.Index, tuple.close)
            prev_ema.append(float(self.ema.ToString()))
        
        rsi_data = pd.DataFrame(prev_rsi, columns = ["rsi"])
        macd_data = pd.DataFrame(prev_macd, columns = ["macd"])
        up_bb_data = pd.DataFrame(up_bb, columns = ["up_bb"])
        low_bb_data = pd.DataFrame(low_bb, columns = ["low_bb"])
        sd_bb_data = pd.DataFrame(sd_bb, columns = ["sd_bb"])
        stochastic_data = pd.DataFrame(prev_stochastic, columns = ["stochastic"])
        ema_data = pd.DataFrame(prev_ema, columns = ["ema"])


        self.indicators_data = pd.concat([rsi_data, macd_data, up_bb_data, low_bb_data, sd_bb_data, stochastic_data, ema_data], axis=1)
        self.indicators_data = self.indicators_data.iloc[20:]
        self.indicators_data.reset_index(inplace=True, drop=True)


        self.currency_data.reset_index(level = [0, 1], drop = True, inplace = True)
        
        self.currency_data.drop(columns=["askopen", "askhigh", "asklow", "askclose", "bidopen", "bidhigh", "bidhigh", "bidlow", "bidclose"], inplace=True)
        self.currency_data = self.currency_data.iloc[20:]
        self.currency_data.reset_index(inplace=True, drop=True)


        close_prev_prices = self.previous_prices("close", self.currency_data["close"], 6)
        open_prev_prices = self.previous_prices("open", self.currency_data["open"], 6)
        high_prev_prices = self.previous_prices("high", self.currency_data["high"], 6)
        low_prev_prices = self.previous_prices("low", self.currency_data["low"], 6)
        
        all_prev_prices = pd.concat([close_prev_prices, open_prev_prices, high_prev_prices, low_prev_prices], axis=1)
        
        final_table = self.currency_data.join(all_prev_prices, how="outer")
        final_table = final_table.join(self.indicators_data, how="outer")
        
        # Drop NaN from feature table
        self.features = final_table.dropna()
        
        self.features.reset_index(inplace=True, drop=True)
        
        # Make labels
        self.labels = self.features["close"]
        self.labels = pd.DataFrame(self.labels)
        self.labels.index -= 1
        self.labels = self.labels[1:]
        new_row = pd.DataFrame({"close": [y_close]})
        self.labels = self.labels.append(new_row)
        self.labels.reset_index(inplace=True, drop=True)
        
        
        ## Define scaler for this class
        self.scaler_X = MinMaxScaler()
        self.scaler_X.fit(self.features)
        self.scaled_features = self.scaler_X.transform(self.features)
        
        self.scaler_Y = MinMaxScaler()
        self.scaler_Y.fit(self.labels)
        self.scaled_labels = self.scaler_Y.transform(self.labels)
        
        ## fine tune the model to determine hyperparameters
        ## only done once (upon inititialize)
        
        tscv = TimeSeriesSplit(n_splits=2)
        cells = [100, 200]
        epochs = [100, 200]
        
        ## create dataframee to store optimal hyperparams
        params_data = pd.DataFrame(columns = ["cells", "epoch", "mse"])
        
        
        # # Extract the optimised values of cells and epochcs from abbove row (having min mse)
        self.opt_cells = 200
        self.opt_epochs = 100
        # self.opt_cells = O_values["cells"][0]
        # self.opt_epochs = O_values["epoch"][0]
        
        
        X_train = np.reshape(self.scaled_features, (self.scaled_features.shape[0], 1, self.scaled_features.shape[1]))
        y_train = self.scaled_labels
        
        
        self.session = K.get_session()
        self.graph = tf.get_default_graph()

        # Intialise the model with optimised parameters
        self.model = Sequential()
        self.model.add(LSTM(self.opt_cells, input_shape = (1, X_train.shape[2]), return_sequences = True))
        self.model.add(Dropout(0.20))
        self.model.add(LSTM(self.opt_cells,return_sequences = True))
        self.model.add(Dropout(0.20))
        self.model.add(LSTM(self.opt_cells, return_sequences = True))
        self.model.add(LSTM(self.opt_cells))
        self.model.add(Dropout(0.20))
        self.model.add(Dense(1))
        
        self.model.compile(loss= 'mean_squared_error',optimizer = 'adam', metrics = ['mean_squared_error'])
        


    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''
        current_price = data[self.currency].Price
        
        ## call in previous 1 day data
        y_data = self.History([self.currency,], 1, Resolution.Daily)
        
        if y_data.empty:
            y_data = self.History([self.currency,], 2, Resolution.Daily)

        ### Update the Features and Labels for Fitting ###
        
        features_t_minus_1 = self.features[-6:]
        
        ## generate prev 6 datapoints (as features)
        close_prev_prices = self.previous_prices("close", features_t_minus_1["close"], 6)
        open_prev_prices = self.previous_prices("open", features_t_minus_1["open"], 6)
        high_prev_prices = self.previous_prices("high", features_t_minus_1["high"], 6)
        low_prev_prices = self.previous_prices("low", features_t_minus_1["low"], 6)
        
        ## join all
        all_prev_prices = pd.concat([close_prev_prices, open_prev_prices, high_prev_prices, low_prev_prices], axis=1)
        all_prev_prices.reset_index(drop=True, inplace=True)
        
        
        ## get the indicators
        prev_stochastic, prev_rsi, prev_bb, low_bb, up_bb, prev_macd, sd_bb, prev_ema = [],[],[],[],[],[],[],[]
        
        for tuple in y_data.loc[self.currency].itertuples():        
            # making Ibasedatabar for stochastic
            bar = QuoteBar(tuple.Index, 
                           self.currency,
                           Bar(tuple.bidclose, tuple.bidhigh, tuple.bidlow, tuple.bidopen),
                           0,
                           Bar(tuple.askclose, tuple.askhigh, tuple.asklow, tuple.askopen),
                           0,
                           timedelta(days=1)
                          )
            
            self.stochastic.Update(bar)
            prev_stochastic.append(float(self.stochastic.ToString()))
            
            self.rsi.Update(tuple.Index, tuple.close)
            prev_rsi.append(float(self.rsi.ToString()))
            
            self.bb.Update(tuple.Index, tuple.close)
            prev_bb.append(float(self.bb.ToString()))
            low_bb.append(float(self.bb.LowerBand.ToString()))
            up_bb.append(float(self.bb.UpperBand.ToString()))
            sd_bb.append(float(self.bb.StandardDeviation.ToString()))
            
            self.macd.Update(tuple.Index, tuple.close)
            prev_macd.append(float(self.macd.ToString()))
            
            self.ema.Update(tuple.Index, tuple.close)
            prev_ema.append(float(self.ema.ToString()))
            
        
        rsi_data = pd.DataFrame(prev_rsi, columns = ["rsi"])
        macd_data = pd.DataFrame(prev_macd, columns = ["macd"])
        up_bb_data = pd.DataFrame(up_bb, columns = ["up_bb"])
        low_bb_data = pd.DataFrame(low_bb, columns = ["low_bb"])
        sd_bb_data = pd.DataFrame(sd_bb, columns = ["sd_bb"])
        stochastic_data = pd.DataFrame(prev_stochastic, columns = ["stochastic"])
        ema_data = pd.DataFrame(prev_ema, columns = ["ema"])
        
        indicators_data = pd.concat([rsi_data, macd_data, up_bb_data, low_bb_data, sd_bb_data, stochastic_data, ema_data], axis=1)
        indicators_data.reset_index(inplace=True, drop=True)
        
        y_data.drop(columns=["askopen", "askhigh", "asklow", "askclose", "bidopen", "bidhigh", "bidhigh", "bidlow", "bidclose"], inplace=True)
        y_data.reset_index(drop=True, inplace=True)
        
        y_data = y_data.join(all_prev_prices, how="outer")
        y_data = y_data.join(indicators_data, how="outer")
        
        # Fit the model at every onData
        with self.session.as_default():
            with self.graph.as_default():
                # self.model.fit(X_train, y_train, epochs=self.opt_epochs, verbose=0)
        
        
                # Get prediction for y_data instance to predict T+1
                # scaler_X.fit(y_data)
                self.Debug(str(y_data))
                scaled_y_data = self.scaler_X.transform(y_data)
                X_predict = np.reshape(scaled_y_data, (scaled_y_data.shape[0], 1, scaled_y_data.shape[1]))
                
                close_price = self.model.predict_on_batch(X_predict)
                                
        
                close_price_prediction = self.scaler_Y.inverse_transform(close_price)
                close_price_prediction = close_price_prediction[0][0]
                self.Debug(close_price_prediction)
        
        
        ## BUY/SELL STRATEGY BASED ON PREDICTED PRICE
        #Make decision for trading based on the output from LSTM and the current price.
        #If output ( forecast) is greater than current price , we will buy the currency; else, do nothing.
        # Only one trade at a time and therefore made a list " self.long_l". 
        #As long as the currency is in that list, no further buying can be done.
        # Risk and Reward are defined: Ext the trade at 1% loss or 1 % profit.
        # Generally the LSTM model can predict above/below the current price and hence a random value is used
        #to scale it down/up. Here the number is 1.1 but can be backtested and optimised.
        
        # If RSI is below 30, shouldn't 
        if close_price_prediction > current_price and self.currency not in self.long_l and self.currency not in self.short_l:
            
            self.Debug("output is greater")
            # Buy the currency with X% of holding in this case 90%
            self.SetHoldings(self.currency, 0.8)
            self.long_l.append(self.currency)
            self.Debug("long")
            
        if self.currency in self.long_l:
            cost_basis = self.Portfolio[self.currency].AveragePrice
            #self.Debug("cost basis is " +str(cost_basis))
            if  ((current_price <= float(0.99) * float(cost_basis)) or (current_price >= float(1.03) * float(cost_basis))):
                self.Debug("SL-TP reached")
                #self.Debug("price is" + str(price))
                #If true then sell
                self.SetHoldings(self.currency, 0)
                self.long_l.remove(self.currency)
                self.Debug("squared")
        #self.Debug("END: Ondata")
        
        # Short
        if close_price_prediction < current_price and self.currency not in self.short_l and self.currency not in self.long_l:
                
            self.SetHoldings(self.currency, -0.8)
            self.short_l.append(self.currency)
            self.Debug("short")
            
                
        if self.currency in self.short_l:
            cost_basis = self.Portfolio[self.currency].AveragePrice
            #self.Debug("cost basis is " +str(cost_basis))
            if  ((current_price <= float(0.97) * float(cost_basis)) or (current_price >=float(1.01) * float(cost_basis))):
                self.Debug("SL-TP reached")
                #self.Debug("price is" + str(price))
                #If true then sell
                self.SetHoldings(self.currency, 0)
                self.short_l.remove(self.currency)
                self.Debug("squared")
        
                
    

    def previous_prices(self, raw_type, data, num_lookback):
        
        '''
        num_lookback is the number of previous prices
        Data is open, high, low or close
        Data is a series
        Returns a dataframe of previous prices
        '''
            
        prices = []
        length = len(data)
        
        for i in range(num_lookback, length+1):
            this_data = np.array(data[i-num_lookback : i])
            prices.append(this_data)
        
        prices_data = pd.DataFrame(prices)
                
        columns = {}
            
        for index in prices_data.columns:
            columns[index] = "{0}_shifted_by_{1}".format(raw_type, num_lookback - index)
            
        prices_data.rename(columns = columns, inplace=True)
        prices_data.index += num_lookback
            
        return prices_data
        
    def OnEndOfAlgorithm(self):
        """Method called when the algorithm terminates."""

        # liquidate all holdings (all unrealized profits/losses will be realized).
        # long and short positions are closed irrespective of profits/losses.
        self.Liquidate(self.currency)