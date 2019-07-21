# LSTM strategy based on indicators only
import tensorflow as tf

import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics
from keras import backend as K

from datetime import datetime, timedelta

import pandas as pd
from copy import deepcopy

seed = 12345
random.seed(seed)
np.random.seed(seed)

class LSTM01Algorithm(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.session = K.get_session()
        self.graph = tf.get_default_graph()

        self.SetStartDate(2018,7,6)  #Set Start Date
        self.SetEndDate(2019, 5, 6)    #Set End Date
        self.SetCash(100000)           #Set Strategy Cash

        ## start the Keras/ Tensorflow session
        self.session = K.get_session()
        self.graph = tf.get_default_graph()

        ## set the currency pair that we are trading, and the correlated currency pair
        self.currency = "EURUSD"
        self.AddForex(self.currency, Resolution.Daily)

        ## define a long list, short list and portfolio
        self.long_l, self.short_l = [], []


        # Initialise indicators
        self.rsi = RelativeStrengthIndex(9)
        self.bb = BollingerBands(14, 2, 2)
        self.macd = MovingAverageConvergenceDivergence(12, 26, 9)
        self.stochastic = Stochastic(14, 3, 3)
        self.ema = ExponentialMovingAverage(9)

        ## Arrays to store the past indicators
        prev_rsi, prev_bb, prev_macd, low_bb, up_bb, sd_bb, prev_stochastic, prev_ema = [],[],[],[],[],[],[],[]

        ## Make history calls for both currency pairs
        self.currency_data = self.History([self.currency], 150, Resolution.Daily) # Drop the first 20 for indicators to warm up

        ## save the most recent open and close
        ytd_open = self.currency_data["open"][-1]
        ytd_close = self.currency_data["close"][-1]

        ## remove yesterday's data. We will query this onData
        self.currency_data = self.currency_data[:-1]
        # self.correl_data = self.correl_data[:-1]

        ## iterate over past data to update the indicators
        for tuple in self.currency_data.loc[self.currency].itertuples():
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

        ## Forming the Indicators data
        ## This is common to the Price Prediction
        rsi_data = pd.DataFrame(prev_rsi, columns = ["rsi"])
        macd_data = pd.DataFrame(prev_macd, columns = ["macd"])
        up_bb_data = pd.DataFrame(up_bb, columns = ["up_bb"])
        low_bb_data = pd.DataFrame(low_bb, columns = ["low_bb"])
        sd_bb_data = pd.DataFrame(sd_bb, columns = ["sd_bb"])
        stochastic_data = pd.DataFrame(prev_stochastic, columns = ["stochastic"])
        ema_data = pd.DataFrame(prev_ema, columns=["ema"])

        self.indicators_data = pd.concat([rsi_data, macd_data, up_bb_data, low_bb_data, sd_bb_data, stochastic_data, ema_data], axis=1)
        self.indicators_data = self.indicators_data.iloc[20:]
        self.indicators_data.reset_index(inplace=True, drop=True)


        ## Currency Data Price
        self._currency_data = deepcopy(self.currency_data)
        self._currency_data = self._currency_data.reset_index(level = [0, 1], drop = True)

        self._currency_data.drop(columns=["askopen", "askhigh", "asklow", "askclose", "bidopen", "bidhigh", "bidhigh", "bidlow", "bidclose"], inplace=True)
        self._currency_data = self._currency_data.iloc[20:]
        self._currency_data.reset_index(inplace=True, drop=True)


        ## saving the previous 6 days OHLC for the price prediction model
        _close_prev_prices = self._previous_prices("close", self._currency_data["close"], 6)
        _open_prev_prices = self._previous_prices("open", self._currency_data["open"], 6)
        _high_prev_prices = self._previous_prices("high", self._currency_data["high"], 6)
        _low_prev_prices = self._previous_prices("low", self._currency_data["low"], 6)

        _all_prev_prices = pd.concat([_close_prev_prices, _open_prev_prices, _high_prev_prices, _low_prev_prices], axis=1)

        _final_table = self._currency_data.join(_all_prev_prices, how="outer")
        _final_table = _final_table.join(self.indicators_data, how="outer")


        # Drop NaN from feature table
        self._features = _final_table.dropna()

        self._features.reset_index(inplace=True, drop=True)

        # Make labels for LSTM model
        self._labels = self._features["close"]
        self._labels = pd.DataFrame(self._labels)
        self._labels.index -= 1
        self._labels = self._labels[1:]
        _new_row = pd.DataFrame({"close": [ytd_close]})
        self._labels = self._labels.append(_new_row)
        self._labels.reset_index(inplace=True, drop=True)

        # Currency Data Direction
        self.currency_data_direction = self.currency_data.reset_index(level = [0, 1], drop = True)

        self.currency_data_direction.drop(columns=["askopen", "askhigh", "asklow", "askclose", "bidopen", "bidhigh", "bidhigh",
                                    "bidlow", "bidclose", "open", "high", "low"], inplace=True)
        self.currency_data_direction = self.currency_data_direction.iloc[20:]
        self.currency_data_direction.reset_index(inplace=True, drop=True)

        # Close Price Direction Change
        self.close_dir_change = self.direction_change("close", self.currency_data_direction["close"], 11)

        # Join the tables
        joined_table_direction = self.currency_data_direction.join(self.close_dir_change, how="outer")
        joined_table_direction = joined_table_direction.join(self.indicators_data, how="outer")

        # Features Direction
        self.features_direction = joined_table_direction.dropna()
        self.features_direction.reset_index(inplace=True, drop=True)

        ## lowerBB and upperBB should change to the difference
        self.features_direction["low_bb_diff"] = self.features_direction["close"] - self.features_direction["low_bb"]
        self.features_direction["up_bb_diff"] = self.features_direction["up_bb"] - self.features_direction["close"]
        self.features_direction["ema_diff"] = self.features_direction["ema"] - self.features_direction["close"]

        self.features_direction.drop(columns=["up_bb", "low_bb", "ema"], inplace=True)

        # Make raw data for labels

        self.labels = self.features_direction["close"]
        self.labels = pd.DataFrame(self.labels)
        self.labels.index -= 1

        self.labels = self.labels[1:]

        new_row = pd.DataFrame({"close": [ytd_close]})
        self.labels = self.labels.append(new_row)

        self.labels.reset_index(inplace=True, drop=True)

        ## Form the binary labels: 1 for up and 0 for down
        self.labels_direction_new = pd.DataFrame(columns=["direction"])
        for row in self.labels.iterrows():

            new_close, old_close = row[1], self.features_direction["close"][row[0]]
            change = (new_close - old_close)[0]
            percent_change = 100*change/old_close

            if percent_change >=0:
                this_data = pd.DataFrame({"direction":[1]})

            elif percent_change <0:
                this_data = pd.DataFrame({"direction":[0]})

            self.labels_direction_new = self.labels_direction_new.append(this_data)

        self.labels_direction_new.reset_index(inplace=True, drop =True)

        ## Test out different features
        self.features_direction.drop(columns=[ "rsi", "stochastic", "close", "sd_bb"], inplace=True)


        self.scaler_X = MinMaxScaler()
        self.scaler_X.fit(self.features_direction)
        scaled_features_direction = self.scaler_X.transform(self.features_direction)

        # Hyperparameters Funetuning
        max_depth= [10, 15, 20, 30]
        n_estimators= [100, 200, 300, 500]
        criterion= ["gini", "entropy"]

        tscv = TimeSeriesSplit(n_splits=4)

        params_data = pd.DataFrame(columns = ["depth", "n_est", "criterion", "acc_score"])

        for depth in max_depth:
            for n_est in n_estimators:
                for crn in criterion:
                    acc_scores = []
                    for train_index, test_index in tscv.split(scaled_features_direction):
                        X_train, X_test = scaled_features_direction[train_index], scaled_features_direction[test_index]

                        Y_train, Y_test = self.labels_direction_new["direction"][train_index], self.labels_direction_new["direction"][test_index]

                        Y_train, Y_test = Y_train.astype('int'), Y_test.astype('int')

                        RF = RandomForestClassifier(criterion=crn, n_estimators=n_est, max_depth=depth, random_state=12345)
                        RF_model = RF.fit(X_train, Y_train)

                        y_pred = RF_model.predict(X_test)

                        acc_score = accuracy_score(Y_test, y_pred)
                        acc_scores.append(acc_score)

                    average_acc = np.mean(acc_scores)

                    ## make this data for cells, epoch and mse and append to params_data
                    this_data = pd.DataFrame({"depth": [depth], "n_est":[n_est], "criterion": [crn], "acc_score": [average_acc]})
                    params_data = params_data.append(this_data)

        opt_values = params_data[params_data['acc_score'] == params_data['acc_score'].max()]
        opt_depth, opt_n_est, opt_crn = opt_values["depth"][0], opt_values["n_est"][0], opt_values["criterion"][0]

        self.RF = RandomForestClassifier(criterion="gini", n_estimators=300, max_depth=10, random_state=123)
        self.RF_model = self.RF.fit(scaled_features_direction, self.labels_direction_new["direction"].astype('int'))

        ## Define scaler for this class
        self._scaler_X = MinMaxScaler()
        self._scaler_X.fit(self._features)
        self._scaled_features = self._scaler_X.transform(self._features)

        self._scaler_Y = MinMaxScaler()
        self._scaler_Y.fit(self._labels)
        self._scaled_labels = self._scaler_Y.transform(self._labels)

        ## fine tune the model to determine hyperparameters
        ## only done once (upon inititialize)

        _tscv = TimeSeriesSplit(n_splits=2)
        _cells = [100, 200]
        _epochs = [50, 100]

        ## create dataframee to store optimal hyperparams
        _params_data = pd.DataFrame(columns = ["cells", "epoch", "mse"])

        # ## loop thru all combinations of cells and epochs
        for i in _cells:
            for j in _epochs:

                print("CELL", i, "EPOCH", j)

                # list to store the mean square errors
                cvscores = []

                for train_index, test_index in _tscv.split(self._scaled_features):
                    X_train, X_test = self._scaled_features[train_index], self._scaled_features[test_index]
                    Y_train, Y_test = self._scaled_labels[train_index], self._scaled_labels[test_index]

                    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

                    model = Sequential()
                    model.add(LSTM(i, input_shape = (1, X_train.shape[2]), return_sequences = True))
                    model.add(Dropout(0.10))
                    model.add(LSTM(i,return_sequences = True))
                    model.add(LSTM(i))
                    model.add(Dropout(0.10))
                    model.add(Dense(1))
                    model.compile(loss= 'mean_squared_error',optimizer = 'rmsprop', metrics = ['mean_squared_error'])
                    model.fit(X_train,Y_train,epochs=j,verbose=0)

                    scores = model.evaluate(X_test, Y_test)
                    cvscores.append(scores[1])

                ## get average value of mean sq error
                MSE = np.mean(cvscores)

                ## make this data for cells, epoch and mse and append to params_data
                this_data = pd.DataFrame({"cells": [i], "epoch":[j], "mse": [MSE]})

                _params_data = _params_data.append(this_data)
                self.Debug(_params_data)



        # # Check the optimised values (O_values) obtained from cross validation
        # # This code gives the row which has minimum mse and store the values to O_values
        # _O_values = _params_data[_params_data['mse'] == _params_data['mse'].min()]

        # # Extract the optimised values of cells and epochcs from abbove row (having min mse)
        self._opt_cells = 200
        self._opt_epochs = 100


        _X_train = np.reshape(self._scaled_features, (self._scaled_features.shape[0], 1, self._scaled_features.shape[1]))
        _y_train = self._scaled_labels


        self._session = K.get_session()
        self._graph = tf.get_default_graph()

        # Intialise the model with optimised parameters
        self._model = Sequential()
        self._model.add(LSTM(self._opt_cells, input_shape = (1, _X_train.shape[2]), return_sequences = True))
        self._model.add(Dropout(0.20))
        self._model.add(LSTM(self._opt_cells,return_sequences = True))
        self._model.add(Dropout(0.20))
        self._model.add(LSTM(self._opt_cells, return_sequences = True))
        self._model.add(LSTM(self._opt_cells))
        self._model.add(Dropout(0.20))
        self._model.add(Dense(1))

        # self.model.add(Activation("softmax"))
        self._model.compile(loss= 'mean_squared_error',optimizer = 'adam', metrics = ['mean_squared_error'])




    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''

        # Make a historical call for yesterday's prices
        ytd_data = self.History([self.currency,], 1, Resolution.Daily)

        if ytd_data.empty:
            ytd_data = self.History([self.currency,], 2, Resolution.Daily)
            ytd_data.dropna(inplace=True)

        # Features for price prediction
        _features_t_minus_1 = self._features[-6:]

        ## generate prev 6 datapoints (as features for price prediction)
        _close_prev_prices = self._previous_prices("close", _features_t_minus_1["close"], 6)
        _open_prev_prices = self._previous_prices("open", _features_t_minus_1["open"], 6)
        _high_prev_prices = self._previous_prices("high", _features_t_minus_1["high"], 6)
        _low_prev_prices = self._previous_prices("low", _features_t_minus_1["low"], 6)

        ## join all OPHL prices to form the price prediction features
        _all_prev_prices = pd.concat([_close_prev_prices, _open_prev_prices, _high_prev_prices, _low_prev_prices], axis=1)
        _all_prev_prices.reset_index(drop=True, inplace=True)

        ## Update indicators
        ## get the indicators
        prev_stochastic, prev_rsi, prev_bb, low_bb, up_bb, prev_macd, sd_bb, prev_ema = [],[],[],[],[],[],[],[]

        for tuple in ytd_data.loc[self.currency].itertuples():
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

        # Dataframes to store all the indicators
        rsi_data = pd.DataFrame(prev_rsi, columns = ["rsi"])
        macd_data = pd.DataFrame(prev_macd, columns = ["macd"])
        up_bb_data = pd.DataFrame(up_bb, columns = ["up_bb"])
        low_bb_data = pd.DataFrame(low_bb, columns = ["low_bb"])
        sd_bb_data = pd.DataFrame(sd_bb, columns = ["sd_bb"])
        stochastic_data = pd.DataFrame(prev_stochastic, columns = ["stochastic"])
        ema_data = pd.DataFrame(prev_ema, columns = ["ema"])

        indicators_data = pd.concat([rsi_data, macd_data, up_bb_data, low_bb_data, sd_bb_data, stochastic_data, ema_data], axis=1)
        indicators_data.reset_index(inplace=True, drop=True)

        # Price Prediction Model's Yesterday Data
        _ytd_data = deepcopy(ytd_data)
        _ytd_data = _ytd_data.reset_index(drop=True)

        _ytd_data.drop(columns=["askopen", "askhigh", "asklow", "askclose", "bidopen", "bidhigh", "bidhigh", "bidlow", "bidclose"], inplace=True)
        _ytd_data.reset_index(drop=True, inplace=True)

        _ytd_data = _ytd_data.join(_all_prev_prices, how="outer")
        _ytd_data = _ytd_data.join(indicators_data, how="outer")


        ## Direction Prediction Model's Yesterday Data (Drop everything from ytd data so only close price remains)
        ytd_data.drop(columns=["askopen", "askhigh", "asklow", "askclose", "bidopen", "bidhigh", "bidhigh", "bidlow", "bidclose", "open", "high", "low"], inplace=True)
        ytd_data.reset_index(drop=True, inplace=True)


        self.currency_data_direction = self.currency_data_direction.append(ytd_data)
        # self.correl_data = self.correl_data.append(ytd_data_correl)

        curr_price = ytd_data["close"][0]

        # Prediction for Direction
        new_dir_change = self.direction_change("close", self.currency_data_direction[-11:]["close"], 11)

        X_pred = new_dir_change
        X_pred.reset_index(inplace=True, drop=True)
        X_pred = X_pred.join(indicators_data, how = "outer")

        ## lowerBB and upperBB should change to the difference
        X_pred["low_bb_diff"] = ytd_data["close"] - X_pred["low_bb"]
        X_pred["up_bb_diff"] = X_pred["up_bb"] - ytd_data["close"]
        X_pred["ema_diff"] = X_pred["ema"] - ytd_data["close"]

        curr_price - float(self.bb.LowerBand.ToString())

        ## Define the buy signal and sell signal for direction
        buy_sig = (55 < float(self.rsi.ToString()) < 70 and 0<= curr_price - float(self.bb.LowerBand.ToString())  <= 0.02) or float(self.macd.ToString())> 0.001
        sell_sig = (20 < float(self.rsi.ToString()) < 65 and 0<= float(self.bb.UpperBand.ToString()) - curr_price <=0.025) or float(self.macd.ToString())<-0.001

        X_pred.drop(columns=["rsi", "stochastic", "sd_bb", "ema", "low_bb", "up_bb"], inplace=True)
        X_pred.reset_index(inplace=True, drop=True)
        X_pred.dropna(inplace=True)

        X_pred = self.scaler_X.transform(X_pred)

        result = self.RF_model.predict(X_pred)
        direction_pred = result[0]

        # LSTM model for price prediction
        with self._session.as_default():
            with self._graph.as_default():

                _scaled_ytd_data = self._scaler_X.transform(_ytd_data)
                _X_predict = np.reshape(_scaled_ytd_data, (_scaled_ytd_data.shape[0], 1, _scaled_ytd_data.shape[1]))

                _close_price = self._model.predict_on_batch(_X_predict)


                _close_price_prediction = self._scaler_Y.inverse_transform(_close_price)
                _close_price_prediction = _close_price_prediction[0][0]


        # Buy Sell Strategy
        if direction_pred == 0 and (sell_sig or _close_price_prediction <= 0.995*curr_price) and self.currency not in self.short_l and self.currency not in self.long_l:

            self.SetHoldings(self.currency, -0.9)
            self.short_l.append(self.currency)
            self.Debug("short")

        if self.currency in self.short_l:
            cost_basis = self.Portfolio[self.currency].AveragePrice
            if  ((curr_price <= float(0.97) * float(cost_basis)) or (curr_price >= float(1.02) * float(cost_basis))):
                self.Debug("SL-TP reached")
                #If true then sell
                self.SetHoldings(self.currency, 0)
                self.short_l.remove(self.currency)
                self.Debug("squared")

        if direction_pred == 1 and (buy_sig or _close_price_prediction > 1.015*curr_price) and self.currency not in self.short_l and self.currency not in self.long_l and float(self.rsi.ToString())<70:

            self.Debug("output is greater")
            # Buy the currency with X% of holding in this case 90%
            self.SetHoldings(self.currency, 0.5)
            self.long_l.append(self.currency)
            self.Debug("long")

        if self.currency in self.long_l:
            cost_basis = self.Portfolio[self.currency].AveragePrice
            if  ((curr_price <= float(0.98) * float(cost_basis)) or (curr_price >= float(1.03) * float(cost_basis))):
                self.Debug("SL-TP reached")
                #If true then sell
                self.SetHoldings(self.currency, 0)
                self.long_l.remove(self.currency)
                self.Debug("squared")


    def direction_change(self, raw_type, data, num_lookback):

        '''
        num_lookback is the number of previous prices
        raw_type is a string: open, high, low or close
        Data is a series
        Returns a dataframe of previous prices
        '''

        prices = []
        length = len(data)

        for i in range(num_lookback, length+1):
            this_data = np.array(data[i-num_lookback : i])

            # input is change in priice
            prices.append(np.diff(this_data.copy()))

        prices_data = pd.DataFrame(prices)

        columns = {}

        for index in prices_data.columns:
            columns[index] = "{0}_shifted_by_{1}".format(raw_type, num_lookback - index-1)

        prices_data.rename(columns = columns, inplace=True)

        prices_data.index += num_lookback - 1

        return prices_data

    def _previous_prices(self, raw_type, data, num_lookback):

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
