import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from data import Data
# import mplleaflet


class Plot():
    """Plot Chicago map and different datapoints on it."""
    
    def __init__(self, data:Data, data_folder_path:str='.') -> None:
        self.data_folder_path = os.path.join(data_folder_path)
        self.map_pic_path = os.path.join(data_folder_path, 'square_map_-88°05_-87°35_41°6_42°1.png')
        self.map_bounds = {
            'small': [-87.92, -87.51, 41.64, 42.03],
            'big': [-88.0623, -87.3248, 41.5075, 42.2092],
            'square': [-88.05, -87.35, 41.6, 42.1]
        }
        self.data = data
        # plt.rcParams['figure.figsize'] = [20, 20]
        self.fig = plt.figure(figsize=(10, 10), dpi= 100, facecolor='w', edgecolor='k')
        self.ax = self.fig.add_subplot(aspect='equal', adjustable='box')
        # self.ax.set_aspect('equal')
        
        # self.fig, self.ax = plt.subplots(1, 1, figsize=(20, 20))
        self.ax.set_title('Chicago Ride Hailing Demand Prediction')
    
    
    def plot_data(self):
        self.ax.plot(-87.7, 41.8, 'bo--')
        
    
    def plot_grid(self):
        coords = []
        
        # Add squares coordinates to list + last squares closing edges coordinates 
        last_long = round(self.data.square_coords[-1][0][0] + self.data.square_size, 5)
        last_lat = round(self.data.square_coords[0][-1][1] + self.data.square_size, 5)
        
        for x, long in enumerate(self.data.square_coords):
            for y, lat in enumerate(long):
                coords.append(lat)
            coords.append((self.data.square_coords[x][0][0], last_lat))
        for y, lat in enumerate(self.data.square_coords[0]):
            coords.append((last_long, self.data.square_coords[0][y][1]))
        coords.append((last_long, last_lat))
        
        lat, long = zip(*coords)
        self.ax.scatter(lat, long, c='red', marker='+', linewidths=1.5)
        
        corners = [
            (self.data.square_coords[0][0][0], self.data.square_coords[0][0][1]),
            (self.data.square_coords[-1][0][0] + self.data.square_size, self.data.square_coords[-1][0][1]),
            (self.data.square_coords[-1][-1][0] + self.data.square_size, self.data.square_coords[-1][-1][1] + self.data.square_size),
            (self.data.square_coords[0][-1][0], self.data.square_coords[0][-1][1] + self.data.square_size),
        ]
        
        corners_long, corners_lat = zip(*corners)
        self.ax.fill(corners_long, corners_lat, c=(1, 0.3, 0.3, 0.2))
        # self.ax.plot(long, lat, 'bo-')
        
        map_pic = mpimg.imread(self.map_pic_path)
        
        plt.imshow(map_pic, extent=self.map_bounds['square'], alpha=0.8, aspect="auto")
    
    def show_plot(self):
        # img = plt.imread(self.map_pic)
        # self.plot.imshow(img, zorder=0, extent=self.boundaries)
        return self.fig
    
        