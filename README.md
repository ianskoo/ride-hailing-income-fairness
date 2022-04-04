# ride-hailing-income-fairness
Repository showcasing the work accomplished for my Bachelor thesis, "Predicting Ride-Hailing Demand: A Potential Solution For Decreasing the Income Inequality of Drivers"

## First setup

### Python version and package manager

This guide is tested on Linux, and should work on MacOS as well.
Make sure you have a recent Python version and git installed in your system. The project uses Pyenv to install an older version of Python in the system (3.8), as the taxi simulator doesn't work with the latest releases of Python.
<!-- The project uses [Pipenv](https://pipenv.pypa.io/en/latest/) to efficiently manage a virtual environment using Python 3.8, as the taxi simulator has old dependencies that don't work under the latest Python versions.

If you don't have Pipenv but you have Python and Pip installed, input this command in a shell:

`pip install pipenv` -->

### Virtual environment setup

Install [Pyenv](https://github.com/pyenv/pyenv) to install and manage multiple Python versions on your system. It's best to follow their [installation guide](https://github.com/pyenv/pyenv#installation) on GitHub, as there are some system-specific quirks. Remember to install the [Python build dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment) as listed in their installation guide.

Then, install the Python version required by the project with:

```
pyenv install 3.8.13
```


Clone this repository with the following command (the project will be saved in your current working directory):

```
git clone https://github.com/ianskoo/ride-hailing-income-fairness && cd ride-hailing-income-fairness
```

Then, set the Python 3.8.13 as the local interpreter (just to be sure, as it should already be set in the repository):

```
pyenv local 3.8.13
```

Create a new virtual environment for the project:

```
python3 -m venv env
```

Activate the environment:

```
source env/bin/activate
```

### Requirements

Finally, install the project dependencies with:

```
pip install -r requirements.txt
```

### Data sourcing

Unless you only need the simulator with random customer requests generation as [the original](https://github.com/bokae/taxi), you will need a ride-hailing trip dataset with the following columns available:

`['Trip Start Timestamp', 'Pickup Centroid Latitude', 'Pickup Centroid Longitude', 'Dropoff Centroid Latitude', 'Dropoff Centroid Longitude']`

The project uses the transportation network provider trips dataset collected by the city of Chicago, which can be found on [their portal](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p).

Save the dataset as a CSV in the folder `./prediction-model/data/`. 
<!-- _OPTIONAL: You can rename the csv 'trips.csv' and run the script `shrink_csv.py` to extract a new csv from it keeping only the above mentioned columns to reduce space usage and computation time._ -->

The project is now set up and ready.


## Data cleaning and preparation

For this step, you should have at least an additional half of the dataset weight of free space on your disk. If you haven't before, activate the virtual environment with:

```
source env/bin/activate
```

Open the file plots.ipynb with the virtual environment's Jupyter module:

```
jupyter notebook prediction-model/plots.ipynb
```

_If the last step doesn't work, check that you correctly installed the [Python build dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment) for Pyenv; if not, remove Python 3.8.13 with `pyenv uninstall 3.8.13`, remove the env/ folder and start again._
<!-- Alternatively, you can probably use any means of opening a Python notebook (VSCode, System installation of Jupyter), just make sure to select the Python kernel of this virtual environment (3.8.13), found in `env/bin/python`._ -->

At the beginning of the notebook, an instance of the `data.py` class will be created, which will read the trips dataset chunk per chunk (10M lines at a time) while cleaning it, and finally divide it into a train and a simulation set which will be saved as CSVs in the data folder. These subsets will take much less space on the disk, as they only contain the five needed columns and have a bit less rows.  If the dataset is as big as the Chicago one at the time of writing (240M rows in April '22), this will take several hours depending on the system. After each chunk is processed, a dot "." is printed. With a chunk size of 10M rows and a dataset size of 240M rows, the operation will be completed after ca. 24 dots are printed. 

_Note: If you know a quick fix to make this faster, please submit an issue. I know about the dask library, but it would probably require some work to be integrated in place of Pandas._

This is the instantiation of the Data class in the notebook:

```
data_1 = data.Data(
    trips_data_path="data/trips.csv"
    # grid_square_size=0.1,
    latitude_divisions=7,
    use_cached_ml_dataset=True,
    simulation_set_cutoff_date='2021-11-17 00:00:00',
    n_threads=8
    )
```
* `trips_data_path` lets you specify the dataset (relative) path and name;
* `grid_square_size` or `latitude_divisions` (preferred) lets you specify the granularity of the demand prediction grid;
* `use_cached_ml_dataset` lets you use a .pkl cached ML dataset, which will be created after running the Data class for the first time.
* `simulation_set_cutoff_date` lets you choose a date for dividing the dataset into two parts, one for the ML model (training, testing), and one for the simulations.
* `n_threads`: Choose this based on the number of available threads in your system

After the dataset has been read and cleaned, the program will continue by building the ML training set. This step is parallelized on 8 threads by default, and has progress bars for each processed chunk. 

## Model training and tuning

After this last step is finished, you can continue along following the Python notebook to train and compare models, tune hyperparameters and see generalization errors.


## Simulations

The simulations are performed with a modified and improved version of [this taxi simulator](https://github.com/bokae/taxi). The basic instructions in their Github repository apply, but I streamlied the batch simulations process in one single Python script named `ml_dispatcher_comparison_batch_run.py`. In this script there's a dictionary containing a basic configuration, some helper functions, and the "main" code at the end. In here, various combinations of parameters can be modified in the base config dictionary and simulations can be run with them. A copy of the modified configuration will be saved in `simulator/configs/`, so you can inspect all the configurations I have already used for my thesis. It is therefore superfluous to use `generate_configs.py` and `batch_run.sh`.

The simulation results are saved in `simulator/results/` and can be read and analized using (and perhaps slightly adapting) the scripts in `results_and_plots.ipynb`.

