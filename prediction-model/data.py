"""Class to read the dataset, clean it and store it in cache as a pandas pickle."""
import os
from time import time
import gc
import json
import gzip
# import sys
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# from sklearn.model_selection import train_test_split


class Data:
    """Class to read the dataset, clean it and store it in cache as a pandas pickle."""

    def __init__(self,
                 grid_square_size: float = None,
                 latitude_divisions: int = None,
                 data_folder_path: str = 'data',
                 use_cached_taxi_dataset: bool = True,
                 use_cached_ml_dataset: bool = True,
                 ml_df_to_compare_fname=None,
                 debug=False):

        self.data_folder_path = os.path.join(data_folder_path)
        self.square_size = None
        self.seed = 144

        self.labels = {
            'time': 'Trip Start Timestamp',
            'lat_a': 'Pickup Centroid Latitude',
            'long_a': 'Pickup Centroid Longitude',
            'lat_b': 'Dropoff Centroid Latitude',
            'long_b': 'Dropoff Centroid Longitude'
        }

        self.ml_df_labels = {
            'date': np.datetime64,
            'day': int,
            'month': int,
            'weekday': int,
            'weekofyear': int,
            # 'weekday_in_month': int,
            'hour': int,
            'minute': int,
        }


        if not grid_square_size and not latitude_divisions:
            raise Exception("Either latitude divisions or square size must be given!")
    
        self.original_df_size = 0
        self.dropped_rows_cnt = 0
        self.dropped_invalid = 0
        self.df = self.load_taxi_data(use_cached=use_cached_taxi_dataset)
        self.df.sort_values(self.labels['time'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.extract_simulator_dataset()
        
        if grid_square_size:
            self.square_coords = self.bin_city_coordinates(square_size=grid_square_size)
        elif latitude_divisions:
            self.square_coords = self.bin_city_coordinates(lat_divisions=latitude_divisions)

        self.min_date = self.df[self.labels['time']].min()
        self.max_date = self.df[self.labels['time']].max()
        
        # Delete taxi dataframe from memory to unclog it
        del self.df
        gc.collect
        
        # Create and populate machine learning dataframe
        self.ml_df: pd.DataFrame = None
        self.load_ml_data(use_cached=use_cached_ml_dataset)
        
        # Sum counts of duplicated time steps (caused by chunking and multihtreading)
        if self.ml_df.duplicated(keep='first', subset=['date', 'hour', 'minute']).sum() > 0:
            self.ml_df = self.ml_df.groupby(['date', 'day', 'month', 'weekday', 'weekofyear', 'hour', 'minute']).agg('sum').reset_index()
        
        # self.compute_average_demand()
        
        # Compare current ml_df to another if given
        self.compare_dataframes(data_folder_path, ml_df_to_compare_fname)

    def extract_simulator_dataset(self, use_cached_taxi_dataset):
        # Extract test_set for simulator (5% random sample or continuous subset in time)
        self.cutoff_date = '2021-11-17 00:00:00'
        self.test_set = self.df[self.df.loc[:, self.labels['time']] >= self.cutoff_date]
        if not use_cached_taxi_dataset or not os.path.isfile(os.path.join(self.data_folder_path, 'test_trips.csv')):
            self.test_set.to_csv(os.path.join(self.data_folder_path, 'test_trips.csv'))
            
        self.df = self.df[self.df.loc[:, self.labels['time']] < self.cutoff_date]
        if not use_cached_taxi_dataset or not os.path.isfile(os.path.join(self.data_folder_path, 'train_trips.csv')):
            self.df.to_csv(os.path.join(self.data_folder_path, 'train_trips.csv'))

    def compare_dataframes(self, data_folder_path, ml_df_to_compare_fname):
        if ml_df_to_compare_fname:
            ml_df_to_compare = pd.read_pickle(os.path.join(data_folder_path, ml_df_to_compare_fname)).sort_values(
                by=['weekday', 'hour', 'minute'])
            self.ml_df.info()
            ml_df_to_compare.info()

            print('\n', self.ml_df.describe())
            print(ml_df_to_compare.describe())

    def load_ml_data(self, use_cached: bool):
        cached_fname = f"ml_df_nodupl_meteo_holyd_squares_{len(self.square_coords[0])*len(self.square_coords)}_sqsize_{self.square_size}.pkl"
        cached_path = os.path.join(self.data_folder_path, cached_fname)
        if use_cached and os.path.isfile(cached_path):
            self.ml_df = pd.read_pickle(cached_path)
        else:
            # Creating and populating machine learning dataframe reading Chicago data in chunks
            cnt = 1
            self.ml_df = pd.DataFrame(columns=self.ml_df_labels)
            self.add_ml_df_coord_labels()
            for chunk in pd.read_csv(
                'data/train_trips.csv',
                # usecols=self.labels.values(),
                parse_dates=[self.labels['time']],
                on_bad_lines='warn',
                chunksize=15000000,
                # nrows=235000
            ):
                print(f"Counting requests in chunk {cnt}:")
                self.ml_df = pd.concat([self.ml_df, self.parallelize_dataframe(chunk, self.demand_per_timestep_date)])
                print(self.ml_df)
                cnt += 1
            
            # Sum counts of duplicated time steps (caused by chunking and multihtreading)   
            self.ml_df = self.ml_df.groupby(['date', 'day', 'month', 'weekday', 'weekofyear', 'hour', 'minute']).agg('sum').reset_index()
            self.add_meteo_data()
            self.add_holidays_data()
            self.ml_df.to_pickle(cached_path)

    def load_taxi_data(self, use_cached: bool):
        """Load taxi dataset in csv format.
        By default, the file path is given by "./data/trips.csv" and "./data/trips.pkl"
        for the cached version, which is created after first reading the csv.
        The cached version is subsequently used by default: delete the .pkl
        file to use another csv file.
        """
        csv_path = os.path.join(self.data_folder_path, 'trips_shrunk.csv')
        pickle_path = os.path.join(self.data_folder_path, 'clean_trips.pkl')

        cols_datatypes = {
            'Pickup Centroid Latitude': float,
            'Pickup Centroid Longitude': float,
            'Dropoff Centroid Latitude': object,
            'Dropoff Centroid Longitude': float
        }

        # Read cached dataframe if present or read csv and cache it
        if use_cached and os.path.isfile(os.path.join(pickle_path)):
            data = pd.read_pickle(pickle_path)
        else:
            data = []
            print("Reading CSV...")
            for chunk in pd.read_csv(
                csv_path,
                # usecols=self.labels.values(),
                dtype=cols_datatypes,
                parse_dates=[self.labels['time']],
                on_bad_lines='warn',
                chunksize=10000000,
                # nrows=235000
            ):
                data.append(self.clean_data_cols(chunk))
            data = pd.concat(data)
            print(f"Dropped {self.dropped_rows_cnt} rows containing NaN values, equal to {(self.dropped_rows_cnt / self.original_df_size * 100):2f}%")
            print(f"Dropped {self.dropped_invalid} rows containing invalid coordinates")
            self.cache_dataframe(dataframe=data, fname='clean_trips.pkl')
        return data

    def cache_dataframe(self, dataframe: pd.DataFrame, fname: str):
        """Save dataframe to pandas cache (pickle)"""
        dataframe.to_pickle(os.path.join(self.data_folder_path, fname))

    def clean_data_cols(self, df: pd.DataFrame):
        """Check and clean dataset columns, changes saved to pickled dataframe (not csv)"""
        reg = r"[^\d\.-]"
        # df.info()
        original_size = df.shape[0]
        self.original_df_size += original_size
        print("Cleaning, sorting, and caching dataset...")

        # Drop rows without coordinates
        df.dropna(inplace=True)
        dropped_cnt = original_size - df.shape[0]
        # print(f"Dropped {dropped_cnt} rows containing NaN values, equal to {dropped_cnt / original_size * 100 :2f}%")

        # Drop rows with invalid float coordinates
        for label in list(self.labels.values())[1:]:
            df.drop(index=df[df[label].astype(str).str.count(reg) > 0].index, inplace=True)
        # print(f"Dropped {original_size - df.shape[0] - dropped_cnt} rows containing invalid coordinates")
        self.dropped_invalid += original_size - df.shape[0] - dropped_cnt
        # dropped_cnt = original_size - df.shape[0]

        # Changing datatype of column to float
        for label in list(self.labels.values())[1:]:
            df.loc[:, label] = pd.to_numeric(df.loc[:, label], downcast='float')

        # Sort by datetime
        df.sort_values(self.labels['time'], inplace=True)

        # # Print changes
        # print(f"Dataset cleaned: Dropped {original_size - df.shape[0]} total rows, ")
        # print(f"\requal to {(1 - df.shape[0] / original_size):2f}%")

        # Save changes (not here when using chunks)
        # self.cache_dataframe(dataframe=df, fname='clean_trips.pkl')
        return df

    def bin_city_coordinates(self, lat_divisions=None, square_size=None) -> list:
        """Bin city coordinates into square zones"""
        
        bounds = {
            'south_lat': self.df[self.labels['lat_a']].min(),
            'north_lat': self.df[self.labels['lat_a']].max(),
            'west_long': self.df[self.labels['long_a']].min(),
            'east_long': self.df[self.labels['long_a']].max()
        }
        
        if lat_divisions:
            # Calculate square size based on number of desired latitude divisions
            self.square_size = round((bounds['north_lat'] - bounds['south_lat']) / lat_divisions, 2)
            print(f"City square size: {self.square_size}°")
        elif square_size:
            self.square_size = square_size
        else:
            raise Exception("Either latitude divisions or square size must be given!")

        # If square size bigger than biggest difference in lat or long, set it as that difference + 0.01 (for round err)
        if (self.square_size > (bounds['north_lat'] - bounds['south_lat']) or
                self.square_size > (bounds['east_long'] - bounds['west_long'])):
            self.square_size = round(max((bounds['north_lat'] - bounds['south_lat']),
                                         (bounds['east_long'] - bounds['west_long'])) + 0.01, 3)
            print(f'Given square size too big, decreased to {self.square_size} (1 square for the whole city area)')

        # Build dict of squares (tuples of coordinates), set all to 0/None
        coords = []
        for x, long_step in enumerate(np.arange(bounds['west_long'], bounds['east_long'], self.square_size)):
            coords.append([])
            for y, lat_step in enumerate(
                    np.arange(bounds['south_lat'], bounds['north_lat'], self.square_size)):
                coords[x].append((round(long_step, 5), (round(lat_step, 5))))

        print(f"Number of squares: {len(coords) * len(coords[0])}")

        # print(f"Number of non-empty squares: {len(coords)}")
        return coords
    
    def add_ml_df_coord_labels(self):
        # Add labels for machine learning dataframe
        for x, lat in enumerate(self.square_coords):
            for y, long in enumerate(self.square_coords[x]):
                self.ml_df_labels[str(self.square_coords[x][y])] = int
                # ml_df.insert(len(ml_df.columns), str(coord), 0)

    def demand_per_timestep_weekday(self, input_df=None):
        """Populate ML dataframe with time and square_coords as columns with count of
        ride pickups in each city square, for each distinct weeday and time step (15')."""

        ml_df = pd.DataFrame(columns=self.ml_df_labels)

        # if input_df == None:
        #     input_df = self.df

        # lst_datetime = None
        new_entry = pd.Series(0, index=self.ml_df_labels)

        for row in tqdm(input_df.values):
            # Check if time already present in the dataset, else add row
            ml_row = ml_df[(ml_df['weekday'] == row[1].day_of_week) & (ml_df['hour'] == row[1].hour) & (
                        ml_df['minute'] == row[1].minute)]
            if ml_row.empty:
                new_entry['weekday'] = row[1].day_of_week
                new_entry['hour'] = row[1].hour
                new_entry['minute'] = row[1].minute
                ml_df.loc[ml_df.shape[0]] = new_entry

            # Assign pickup location to city square
            row_coords = np.array([row[2], row[1]])
            # this_sq_coords = np.round(row_coords - row_coords % self.square_size, 5) # modulo method
            this_sq_index = ((row_coords - self.square_coords[0]) // self.square_size)[0]
            try:
                this_sq_coords = self.square_coords[int(this_sq_index[0])][int(this_sq_index[1])]  # division method
            except IndexError:
                print(
                    f'index error! index: {this_sq_index}, container:{self.square_coords}, row coordinates: {row_coords}')
                return
            ml_row_idx = np.where((ml_df['weekday'] == row[1].day_of_week) &
                                  (ml_df['hour'] == row[1].hour) &
                                  (ml_df['minute'] == row[1].minute))[0][0]

            ml_df.iloc[ml_row_idx][str((this_sq_coords[0], this_sq_coords[1]))] += 1
            # print(ml_df)
            # sys.stdout.write(f"\033[{len(ml_df)+1}F")
            # print(f"[{row.index}]", end='\r')

        # print(ml_df)
        return ml_df

    def demand_per_timestep_date(self, input_df=None):
        """Populate ML dataframe with time and square_coords as columns with count of
        ride pickups in each city square, for each distinct date and time step (15')."""

        ml_df = pd.DataFrame(columns=self.ml_df_labels)
        new_entry = pd.Series(0, index=self.ml_df_labels)

        for row in tqdm(input_df.values):
            # Check if time already present in the dataset, else add row
            ml_row = ml_df[
                (ml_df['date'] == row[1]._date_repr) &
                # (ml_df['weekday'] == row[1].day_of_week) &
                (ml_df['hour'] == row[1].hour) &
                (ml_df['minute'] == row[1].minute)
                ]
            if ml_row.empty:
                new_entry['date'] = row[1]._date_repr
                new_entry['day'] = row[1].day
                new_entry['month'] = row[1].month
                new_entry['weekofyear'] = row[1].weekofyear
                new_entry['weekday'] = row[1].day_of_week
                # new_entry['weekday_in_month'] = row[1].day // 7
                new_entry['hour'] = row[1].hour
                new_entry['minute'] = row[1].minute
                ml_df.loc[ml_df.shape[0]] = new_entry

            # Assign pickup location to city square
            row_coords = np.array([row[3], row[2]])
            # this_sq_coords = np.round(row_coords - row_coords % self.square_size, 5) # modulo method
            this_sq_index = ((row_coords - self.square_coords[0]) // self.square_size)[0]
            try:
                this_sq_coords = self.square_coords[int(this_sq_index[0])][int(this_sq_index[1])]  # division method
            except IndexError:
                print(
                    f'index error! index: {this_sq_index}, container:{self.square_coords}, row coordinates: {row_coords}')
                return
            ml_row_idx = np.where((ml_df['date'] == row[1]._date_repr) &
                                  #   (ml_df['weekday'] == row[1].day_of_week) &
                                  (ml_df['hour'] == row[1].hour) &
                                  (ml_df['minute'] == row[1].minute))[0][0]

            ml_df.loc[ml_row_idx, str((this_sq_coords[0], this_sq_coords[1]))] += 1
            # print(ml_df)
            # sys.stdout.write(f"\033[{len(ml_df)+1}F")
            # print(f"[{row.index}]", end='\r')
        return ml_df
    
    # def demand_per_timestep_efficient(self, input_df=None):
    #     timesteps = input_df[self.labels['time']].unique()
    #     for timestep in timesteps:
    #         for square in self.square_coords

    def parallelize_dataframe(self, data, func, n_cores=10):
        # ml_df = pd.DataFrame(columns=self.ml_df_labels)
        
        # Divide dataset equally
        # self.df.sort_values(self.labels['time'], inplace=True)
        df_split = np.array_split(data, n_cores)

        # Start and assign threads, concatenate resulting dataframes
        pool = Pool(len(df_split))
        print(f'Counting requests per square into ML dataframe...')
        ml_df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
            
        return ml_df

    def compute_average_demand(self):
        for i, row in enumerate(self.ml_df.values):
            # Calculate total requests for this time step
            counts = row[7:]
            total = counts.sum()
            # Compute averages and reassign to dataframe
            for j in range(len(counts)):
                counts[j] = round(counts[j] / total, 2)
            # self.ml_df.iloc[i, 7:] = counts
            # row[7:] = counts

    def add_meteo_data(self):
        """Add meteo features to ml dataset"""
        # Read meteo csv
        csv_path = os.path.join(self.data_folder_path, 'meteo_data.csv')
        meteo_df = pd.read_csv(csv_path,
                               skiprows=1,
                               parse_dates=['Date'],
                               usecols=[0, 2, 3, 4])

        # Select range of dates equal to taxi data
        meteo_df = meteo_df[(meteo_df['Date'] >= self.min_date) &
                            (meteo_df['Date'] <= self.max_date)]

        # Fill missing value with average of neighboring values
        # df.interpolate() doesn't work with date encoding, use in subset
        temp = meteo_df.iloc[:, 1:]
        temp.interpolate(method='linear', axis=0, inplace=True)
        # print(temp.compare(meteo_df.iloc[:, 1:]))
        meteo_df.iloc[:, 1:] = temp
        # print(pd.isna(meteo_df).sum())

        meteo_df['Date'] = meteo_df['Date'].astype('str')
        temp2 = pd.merge(meteo_df, self.ml_df, left_on='Date', right_on='date')
        self.ml_df = temp2.drop('date', axis=1)

    def add_holidays_data(self):
        """Add "is_holiday" boolean feature to dataset."""
        csv_path = os.path.join(self.data_folder_path, 'holidays.csv')
        holidays_df = pd.read_csv(csv_path, parse_dates=['date'])

        # Select range of dates equal to taxi data
        holidays_df = holidays_df[(holidays_df['date'] >= self.min_date) &
                                  (holidays_df['date'] <= self.max_date)]
        holidays_df['is_holiday'] = 1

        # Merge with taxi dataset
        holidays_df['date'] = holidays_df['date'].astype('str')
        self.ml_df = pd.merge(holidays_df, self.ml_df, how='right', left_on='date', right_on='Date')
        self.ml_df['is_holiday'] = self.ml_df['is_holiday'].fillna(0)
        self.ml_df['date'] = self.ml_df['Date']
        self.ml_df = self.ml_df.drop('Date', axis=1)
        
