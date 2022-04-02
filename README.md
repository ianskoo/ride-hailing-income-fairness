# ride-hailing-income-fairness
Repository showcasing the work accomplished for my Bachelor thesis, "Predicting Ride-Hailing Demand: A Potential Solution For Decreasing the Income Inequality of Drivers"

## First setup

### Python version and package manager

This guide is tested on Linux, and should work as-is on MacOS as well.
Make sure you have a recent Python version and git installed in your system. The project uses Pyenv to install an older version of Python in the system (3.8), as the taxi simulator doesn't work with the latest releases of Python. If you already have Python 3.8 on your system, you can skip to "Requirements".
<!-- The project uses [Pipenv](https://pipenv.pypa.io/en/latest/) to efficiently manage a virtual environment using Python 3.8, as the taxi simulator has old dependencies that don't work under the latest Python versions.

If you don't have Pipenv but you have Python and Pip installed, input this command in a shell:

`pip install pipenv` -->

### Virtual environment setup

Install [Pyenv](https://github.com/pyenv/pyenv#installation) to install and manage multiple Python versions on your system. It's best to follow their [installation guide](https://github.com/pyenv/pyenv#installation) on GitHub, as there are some system-specific quirks. Remember to installed the Python build dependencies as listed on their guide.

Then, install the Python version required by the project with:

`pyenv install 3.8.13`


Clone this repository with the following command (the project will be saved in your current working directory):

`git clone https://github.com/ianskoo/ride-hailing-income-fairness && cd ride-hailing-income-fairness`

Then, set the Python 3.8 as the local interpreter:

`pyenv local 3.8.13`

Create a new virtual environment for the project:

`python3 -m venv env`

Activate the environment:

`source env/bin/activate`

### Requirements

Finally, install the project dependencies with:

`pip install -r requirements.txt`

### Data sourcing

Unless you only need the simulator with random customer requests generation as [the original](https://github.com/bokae/taxi), you will need a ride-hailing trip dataset with the following columns available:

`['Trip Start Timestamp', 'Pickup Centroid Latitude', 'Pickup Centroid Longitude', 'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude']`

The project uses the transportation network provider trips dataset collected by the city of Chicago, which can be found on [their portal](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p).

Save the dataset as a CSV in the folder `./prediction-model/data/`. 
<!-- _OPTIONAL: You can rename the csv 'trips.csv' and run the script `shrink_csv.py` to extract a new csv from it keeping only the above mentioned columns to reduce space usage and computation time._ -->

The project is now set up and ready.


## Data cleaning and preparation

For this step, you should have at least an additional half of the dataset weight of free space on your disk.

Open the file plots.ipynb with the virtual environment's Jupyter module:

`jupyter notebook prediction-model/plots.ipynb`

If this shouldn't work, you can use any means of opening a Python notebook (VSCode, System installation of Jupyter), just make sure to select the Python kernel of this virtual environment (3.8.13), found in `env/bin/python`.

At the beginning of the notebook, an instance of the `data.py` class will be created, which will read the trips dataset chunk per chunk, clean it, and divide it into a train and a simulation set, which will be saved as CSVs in the data folder. These subsets will take much less space on the disk, as they only contain the five needed columns and have a bit less rows.

This is the instantiation of the Data class in the notebook:

```
data_1 = data.Data(
    trips_data_path="data/trips.csv"
    # grid_square_size=0.1,
    latitude_divisions=7,
    use_cached_ml_dataset=False,
    simulation_set_cutoff_date='2021-11-17 00:00:00',
    )
```
* `trips_data_path` lets you specify the dataset (relative) path and name;
* `grid_square_size` or `latitude_divisions` (preferred) lets you specify the granularity of the demand prediction grid;
* `use_cached_ml_dataset` lets you use a .pkl cached ML dataset, which will be created after running the Data class for the first time.




## Prediction model

