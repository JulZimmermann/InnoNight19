import pandas as pd

class DataFactorizer:

    factDict = {}

    def factorizeDataSave(self, data):
        for attr, v in data.dtypes.items():
            if v == "object":
                self.factDict[attr] = pd.factorize(data[attr])[0]
                data[attr] = self.factDict[attr]

    def factorizeDataLoad(self, data):
        for attr, v in data.dtypes.items():
            if v == "object":
                data[attr] = self.factDict[attr]
