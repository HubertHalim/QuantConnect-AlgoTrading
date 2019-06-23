# This strategy make use of 3 very commonly used indicators: MACD, RSI and Bollinger Bands
# The main indicator is BB - Buy only when the price is 2 SD below or 1 SD above
# If the above condition is met, then the algorithm checks for the basic MACD and RSI strategy

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
import numpy as np

class BasicMACDForexAlgorithm(QCAlgorithm):

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
        self.resolution = Resolution.Daily # set to daily for testing as it takes too very to run minute
        self.AddForex(self.currency,self.resolution)

        # set indicators
        self.macd = self.MACD(self.currency, 12, 26, 9, self.resolution)
        self.rsi = self.RSI(self.currency, 10, self.resolution)
        self.bb1 = self.BB(self.currency, 20, 1, self.resolution)
        self.bb2 = self.BB(self.currency, 20, 2, self.resolution)

        # set warm up period (optional)

        # set reality modelling (optional)
        self.SetBrokerageModel(BrokerageName.Default, AccountType.Cash)

        # set benchmark (optional)

        # set timezone (default is New York time)

        # set additional parameters
        self.previous = None


    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data
        point will be pumped in here. data is a Slice object keyed by symbol containing
        the stock data'''

        # wait for our MACD to fully initialize
        if not self.macd.IsReady:
            return

        # ensure that the algorithm only trades once per day
        if self.previous is not None and self.previous.date() == self.Time.date():
            return

        # define a small tolerance on our checks to avoid bouncing
        tolerance = 0.00015

        holdings = self.Portfolio["EURUSD"].Quantity
        price = self.Securities["EURUSD"].Price

        # to determine variation of macd from the signal line
        MACD_variation = (self.macd.Current.Value - self.macd.Signal.Current.Value)/self.macd.Fast.Current.Value

        # set buy condition
        # buy if price is at bb2's lower band and satisfy either the MACD or RSI criteria
        if holdings <= 0 and price <= self.bb2.LowerBand.Current.Value:
            if MACD_variation > tolerance or self.rsi.Current.Value < 30:
                self.SetHoldings(self.currency, 1.0) # buy EURUSD with all our holdings

        # set sell condition
        # sell if if price is at bb1's upper band and satisfy either the MACD or RSI criteria
        elif holdings >= 0 and price >= self.bb1.UpperBand.Current.Value
            if MACD_variation < -tolerance or self.rsi.Current.Value > 70:
            self.Liquidate(self.currency) # sell all EURUSD holdings

        self.previous = self.Time # updates the previous time

    # for plotting
    def OnEndOfDay(self):
        self.Plot("RSI", "RSI", self.rsi.Current.Value)
        self.Plot("BB", "MiddleBand", self.bb1.MiddleBand.Current.Value)
        self.Plot("BB", "UpperBand", self.bb1.UpperBand.Current.Value)
        self.Plot("BB", "LowerBand", self.bb2.LowerBand.Current.Value)
