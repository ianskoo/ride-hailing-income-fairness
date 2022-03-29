import pickle
from pandas import to_datetime
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Prediction:
    """Class to read a cached sklearn random forest model in pickle format."""

    def __init__(self, model_path: str):
        try:
            self.model = pickle.load(open(model_path, "rb"))
        except FileNotFoundError:
            print("No ML model found within the given path.")
            return

    def predict(self,
                date_time: datetime,
                is_holiday: bool,
                temp_range_f: tuple,
                day_rain_inches: float):
        """

        Parameters
        ----------
        date_time:    A datetime object of the desired prediction date and time
        is_holiday:             Boolean describing if the desired prediction day is a holiday
        temp_range_f:           Tuple of floats for max and min forecasted temperatures in F for the day
        day_rain_inches:        Float with forecasted inches of rain for the day

        Returns
        -------
        predictions:            List of ints for predicted counts of customers per city square
        """

        # Convert python datetime to pandas timestamp
        date_time = to_datetime(date_time)

        # Build data array
        x = [[
            is_holiday,
            temp_range_f[0],
            temp_range_f[1],
            day_rain_inches,
            date_time.day,
            date_time.month,
            date_time.dayofweek,
            date_time.weekofyear,
            date_time.week,
            date_time.hour,
            date_time.minute,
        ]]

        # Scale features and predict
        scaler = MinMaxScaler()
        x_fit = scaler.fit_transform(x)
        return self.model.predict(x_fit)
