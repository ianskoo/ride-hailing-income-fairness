import os
from datetime import date, timedelta, datetime, time
from os.path import join
from random import randint
import pandas as pd
import numpy as np
from geopy import distance

from prediction_model import Prediction


class Chicago:
    """Represent the Chicago test trips data to have real-life requests in the simulation."""

    def __init__(self, 
                 dataset_path: str, 
                 start_date: str,
                 end_date: str,
                 work_hrs_1: tuple,
                 work_hrs_2: tuple,
                 debug=False):

        self.prediction = None
        self.debug = debug
        self.data_folder_path = "data"
        self.labels = {
            'time': 'Trip Start Timestamp',
            'lat_a': 'Pickup Centroid Latitude',
            'long_a': 'Pickup Centroid Longitude',
            'lat_b': 'Dropoff Centroid Latitude',
            'long_b': 'Dropoff Centroid Longitude'
        }
                
        print('Reading trips csv...')
        self.df = pd.read_csv(dataset_path, 
                              parse_dates=[self.labels['time']], 
                              usecols=self.labels.values())
                            #   index_col=[self.labels['time']])
        
        # Select requests spanning desired times only
        self.df = self.prepare_simulation_time_frame(start_date, end_date, work_hrs_1, work_hrs_2)
        self.min_date = self.df[self.labels['time']].min()
        self.max_date = self.df[self.labels['time']].max()

        # Add weather and holiday columns
        self.add_holidays_data(work_hrs_1[0])
        self.add_meteo_data()

        # Stats and prints of simulation time
        self.unique_datetimes = self.df[self.labels['time']].unique()
        print(f"Simulation start datetime: {self.min_date}, end datetime: {self.max_date}")
        tot_tu = self.unique_datetimes.shape[0] * 90
        print(f"Simulation data prepared for {tot_tu} simulation time units.")
                
        # Preassign each request to a tu from 0-89
        self.preassign_requests_to_tu()
        
        # Search for minimum latitudes and longitudes for the city bounds
        self.bounds = {
            'long': (
                min(self.df[self.labels['long_a']].min(), self.df[self.labels['long_b']].min()), 
                max(self.df[self.labels['long_a']].max(), self.df[self.labels['long_b']].max()),
                ),
            'lat': (
                min(self.df[self.labels['lat_a']].min(), self.df[self.labels['lat_b']].min()), 
                max(self.df[self.labels['lat_a']].max(), self.df[self.labels['lat_b']].max()),
                )
        }

        self.du = 100  # simulator distance unit
        self.city_size = self.compute_city_size()
        print(f"City size: {self.city_size}")

        self.n, self.m = self.compute_grid_size()
        # self.m, self.n = self.compute_grid_size()
        print(f"Grid size: {self.n, self.m}")

    def compute_city_size(self):
        """Compute size of city in meters. Returns: (delta longitude, delta latitude)"""
        return (
            distance.distance(((self.bounds['lat'][0]), self.bounds['long'][0]),
                              ((self.bounds['lat'][0]), self.bounds['long'][1])).m,
            distance.distance(((self.bounds['lat'][0]), self.bounds['long'][0]),
                              ((self.bounds['lat'][1]), self.bounds['long'][0])).m
        )

    def compute_grid_size(self):
        return int(np.ceil(self.city_size[0] / self.du)), int(np.ceil(self.city_size[1] / self.du))

    def get_one_request_coord(self, time_unit):
        """
        Gets a time-unit-labeled request, given a specific time step.
        Returns pick-up and drop-off grid coordinates (multiple return)

        Returns
        -------
        int x_a, y_a, x_b, y_b
        """
        # Convert simulator time to datetime and normalize time unit to range (0-89)
        dt = self.sim_time_to_datetime(time_unit)
        norm_time_unit = time_unit % 90

        # Select first request matching normalized time unit
        req_idx = self.df.loc[(self.df[self.labels['time']] == dt) &
                              (self.df['time_unit'] == norm_time_unit)].index[0]
        req = self.df.loc[req_idx]
        # print(req)
        self.df.drop(req_idx, inplace=True)

        # a = chosen_row.loc[self.col_names['lat_a']]
        x_a, y_a = self.coord_to_grid(req[self.labels['lat_a']], req[self.labels['long_a']])
        x_b, y_b = self.coord_to_grid(req[self.labels['lat_b']], req[self.labels['long_b']])

        if self.debug:
            print(f"time unit {time_unit}, datetime {dt}: {y_a, x_a, y_b, x_b}")
            if not time_unit and not time_unit % (90 * 4 * 8 - 1):
                print(f"Starting nr of distinct datetimes: {self.unique_datetimes.shape[0]}, "
                      f"current: {self.df[self.labels['time']].unique().shape[0]}")
        # return y_a, x_a, y_b, x_b
        return x_a, y_a, x_b, y_b

    def coord_to_grid(self, lat: float, long: float):
        """Convert coordinates pair to x, y position in city grid"""

        offset_lat = distance.distance((lat, long), (self.bounds['lat'][0], long)).m
        offset_long = distance.distance((lat, long), (lat, self.bounds['long'][0])).m
        y = int(offset_lat // self.du)
        x = int(offset_long // self.du)

        return x, y
        # return y, x

    def sim_time_to_datetime(self, sim_time):
        """Convert simulation time in time units (tu) to dataset time (datetime)"""
        return self.unique_datetimes[sim_time // 90]

    def prepare_simulation_time_frame(self, 
                                      start_date: str,  # Format yyyy-mm-d
                                      end_date: str,  # Format yyyy-mm-d
                                      work_hrs_1: tuple,
                                      work_hrs_2: tuple):
        """Select data from Chicago dataset, given a specific time frame"""
        sim_df = self.df.set_index(self.labels['time'])
        
        # Iterate over datetime list and build dataframe
        sim_df = sim_df[(sim_df.index >= start_date) &
                        (sim_df.index < end_date)]

        subset_a = sim_df.between_time(work_hrs_1[0].strftime("%H:%M"),
                                       work_hrs_1[1].strftime("%H:%M"),
                                       inclusive='left')
        subset_b = sim_df.between_time(work_hrs_2[0].strftime("%H:%M"),
                                       work_hrs_2[1].strftime("%H:%M"),
                                       inclusive='left')
        sim_df = pd.concat([subset_a, subset_b])
        sim_df.sort_index(inplace=True)
        sim_df.reset_index(inplace=True)
        return sim_df
    
    def preassign_requests_to_tu(self):
        """Preassign requests of a given time step to each simulator time unit (tu) of that time step"""
        
        # Add column for simulation time unit assignment and fill it with random numbers from 0-89
        self.df.insert(self.df.shape[1], 'time_unit', 
                       value=np.random.randint(0, 89, self.df.shape[0]))
        
    def get_request_rate(self, time_unit: int):
        dt = self.sim_time_to_datetime(time_unit)
        request_rate = self.df[(self.df[self.labels['time']] == dt) &
                               (self.df['time_unit'] == time_unit % 90)].shape[0]
        if self.debug:
            print(f"time unit {time_unit} datetime {dt} request rate {request_rate}")
        return request_rate
    
    def add_meteo_data(self):
        """Add meteo features to ml dataset"""
        # Read meteo csv
        csv_path = os.path.join(self.data_folder_path, 'meteo_data.csv')
        meteo_df = pd.read_csv(csv_path,
                               skiprows=1,
                               parse_dates=['Date'],
                               usecols=[0, 2, 3, 4])
        meteo_df['Date'] = meteo_df['Date'].dt.date

        # Fill missing value with average of neighboring values
        # df.interpolate() doesn't work with date encoding, use in subset
        temp = meteo_df.iloc[:, 1:]
        temp.interpolate(method='linear', axis=0, inplace=True)
        meteo_df.iloc[:, 1:] = temp

        # Select range of dates equal to taxi data
        meteo_df = meteo_df[(meteo_df['Date'] >= self.min_date) &
                            (meteo_df['Date'] <= self.max_date)]

        self.df.insert(self.df.shape[1], 'Date', self.df[self.labels['time']].dt.date)
        temp2 = pd.merge(self.df, meteo_df, how='left', on='Date')
        self.df = temp2.drop('Date', axis=1)

    def add_holidays_data(self, start_work_hr):
        """Add "is_holiday" boolean feature to dataset."""
        csv_path = os.path.join(self.data_folder_path, 'holidays.csv')
        holidays_df = pd.read_csv(csv_path, parse_dates=['date'])
        holidays_df['date'] = holidays_df['date'].dt.date
        holidays_df.insert(1, 'is_holiday', 1)

        # Select range of dates equal to taxi data
        holidays_df = holidays_df[(holidays_df['date'] >= self.min_date) &
                                  (holidays_df['date'] <= self.max_date)]

        self.df.insert(self.df.shape[1], 'date', self.df[self.labels['time']].dt.date)
        self.df = pd.merge(self.df, holidays_df, how='left', on='date').drop('date', axis=1)
        self.df['is_holiday'] = self.df['is_holiday'].fillna(0).astype('int')

    def get_future_datetime(self, sim_time: int, offset_mins: int):
        res = pd.to_datetime(self.sim_time_to_datetime(sim_time)) + timedelta(minutes=offset_mins)
        return res

    def get_is_holiday(self, date_time):
        res = self.df[self.df[self.labels['time']] == date_time]
        if res.empty:
            return -1
        res = res.iloc[0, :].loc['is_holiday']
        return res

    def get_temp_range(self, date_time):
        res = self.df[self.df[self.labels['time']] == date_time]
        res = res.iloc[0, :].loc['TMAX (Degrees Fahrenheit)':'TMIN (Degrees Fahrenheit)'].values
        return res

    def get_rain_inches(self, date_time):
        res = self.df[self.df[self.labels['time']] == date_time]
        res = res.iloc[0, :].loc['PRCP (Inches)']
        return res

    def create_chicago_taxi_home_coords(self):
        """
        Take a random request from the whole simulation set and return its coordinates. Avoids drivers being
        spawned on water by the simulation.
        Returns
        -------

        """
        choice = self.df.sample(1)
        choice_coords = choice[self.labels['lat_a']].values[0], choice[self.labels['long_a']].values[0]
        choice_grid_coords = self.coord_to_grid(*choice_coords)
        return choice_grid_coords

    def init_prediction(self, model_path: str):
        self.prediction = Prediction(model_path=model_path)
    
    def predict_demand(self, sim_time: int, offset_mins: int = 30):
        """Predict demand in offset_mins time in all city divisions.
        Return list with counts of predicted customers per city division"""
        # Get features for prediction
        date_time = self.get_future_datetime(sim_time, offset_mins=offset_mins)
        is_holiday = self.get_is_holiday(date_time)

        # If datetime outside working hours (is_holiday returns None), do nothing
        if is_holiday == -1:
            return None

        temp_range_f = self.get_temp_range(date_time)
        day_rain_inches = self.get_rain_inches(date_time)

        # Predict demand in city
        return self.prediction.predict(date_time,
                                       is_holiday,
                                       temp_range_f,
                                       day_rain_inches)
