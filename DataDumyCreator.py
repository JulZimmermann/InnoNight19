import pandas as pd

class DataDumyCreator:

    def createDummies(self, data, yKey, yValidKey):
        for attr, v in data.dtypes.items():
            if v == "object" and attr != yKey:
                dummies = pd.get_dummies(data[attr])

                data = data.join(dummies, rsuffix="_" + attr)
                del data[attr]

            if attr == yKey:
                data[attr] = data[attr].map(lambda x: 1 if x == yValidKey else 0)


        return data
