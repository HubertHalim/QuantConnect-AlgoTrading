# A blank template with only the basic parameters filed in

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
        self.resolution = Resolution.Minute
        self.AddForex(self.currency,self.resolution)

        # set indicators

        # set warm up period (optional)

        # set reality modelling (optional)
        self.SetBrokerageModel(BrokerageName.Default, AccountType.Cash)

        # set benchmark (optional)

        # set timezone (default is New York time)

        # set additional parameters


    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data
        point will be pumped in here. data is a Slice object keyed by symbol containing
        the stock data'''

        # set buy condition

        # set sell condition
