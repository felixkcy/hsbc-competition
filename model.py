import pandas as pd
from fbprophet import Prophet

class EnergyModel:
    def __init__(self):
        self.model = None

    def preprocess_training_data(self, df):
        df.rename(columns={'day': 'ds', 'consumption': 'y'}, inplace=True)
        print(df.columns)
        return df, None
        

    def fit(self, X, y):
        earth_days = pd.DataFrame({
            'holiday': 'earth_days',
            'ds': pd.to_datetime(['2011-04-22', '2012-04-22', '2013-04-21', '2014-04-22'])
        })
        bank_holidays = pd.DataFrame({
            'holiday': 'bank_holidays',
            'ds': pd.to_datetime(list(pd.read_csv('BankHolidayLists.csv')['Date']))
        })
        all_holidays = pd.concat((earth_days, bank_holidays))
        forecast_model = Prophet(growth='linear', weekly_seasonality=3, yearly_seasonality=3, holidays=all_holidays)
        self.model = forecast_model.fit(X)

    def preprocess_unseen_data(self, df):
        df.rename(columns={'day': 'ds'}, inplace=True)
        return df

    def predict(self, X):
	    return self.model.predict(X)['yhat']