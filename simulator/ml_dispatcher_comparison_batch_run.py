import json
import os
import subprocess
import sys

from city_model import Simulation

# Basic simulator settings including Chicago additions
config = {
    "use_chicago": True,
    "chicago_grid_size": 49,
    "start_date": "2020-1-6",
    "end_date": "2020-1-13",
    "working_hrs_1": [8, 10],
    "working_hrs_2": [17, 21],
    "ml_model_path": "data/random_forest_8_300_100_squares_49.pkl",
    "ml_weighted": True,
    "max_time": 15120,
    "batch_size": 15120,
    "num_taxis": 10000,
    "matching": "nearest",
    "behaviour": "stay",
    "initial_conditions": "home",
    "hard_limit": 50,

    "price_fixed": 2,
    "price_per_dist": 1,
    "cost_per_unit": 0.008,
    "log": False,
    "show_map_labels": False,
    "show_pending": False,
    "show_plot": False,
    "max_request_waiting_time": 60,
    "avg_request_lengths": 22.7,
    "request_rate": 1,
    "reset": "false",
    "geom": 0,
    "request_origin_distributions": [{"location": [20, 20], "strength": 1, "sigma": 10}]
}


def change_config(params_to_modify: dict):
    # Set new parameters in config dict
    for key, value in params_to_modify.items():
        config[key] = value


def make_config_files_and_run():
    # Create config folder
    config_simulation_folder_path = f"configs/start_{config['start_date']}_end_{config['end_date']}_" \
                                    f"limit_{config['hard_limit']}_taxis_{config['num_taxis']}/"
    if not os.path.exists(config_simulation_folder_path):
        os.mkdir(config_simulation_folder_path)

    # Create configuration
    config_name = f"matching_{config['matching']}_initial_positions_{config['initial_conditions']}" \
                  f"_behavior_{config['behaviour']}"

    if config['behaviour'] == "ml_dispatcher_v2" and config['ml_weighted']:
        config_name += "_weighted"

    if config['behaviour'] == "ml_dispatcher_v2" or config['behaviour'] == "ml_dispatcher_distributor":
        config_name += f"_prediction_areas_{config['chicago_grid_size']}"

    config_path = config_simulation_folder_path + config_name + '.json'
    with open(config_path, 'w') as file:
        json.dump(config, file)

    run(config, config_path)


def run(run_config, config_path):
    if os.path.exists(config_path):
        p = "/".join(config_path.split("/")[1:][:-1])
        run_id = config_path.split("/")[-1].split(".")[0]
        # config = json.load(open(config_path))

        # Remove results if existing
        # if os.path.exists('results/'+p):
        #     input("Confirm overwriting existing results in 'results/"+p + "'? (Enter to accept or Ctrl-C)")
        #     shutil.rmtree('results/'+p)

        s = Simulation(**run_config)  # create a Simulation instance

        # s.run_batch(run_id, data_path="results/"+p)
        s.run_batch(run_id, data_path=p)
    else:
        print('Please give an existing config file from the "./configs" folder!')


# Customize and run simulations
algorithms = ["nearest", "poorest", "random_limited"]

strats = ["stay", "ml_dispatcher_v2"]  #, "ml_dispatcher_distributor"]
# strats = ["ml_dispatcher_v2", "ml_dispatcher_distributor"]

dates = [("2020-1-13", "2020-1-20"), ("2020-1-20", "2020-1-27"), ("2020-1-27", "2020-2-3")]
# taxis = [10000, 25000]

# for date in dates:
for strat in strats:
    for algo in algorithms:
        # for taxi in taxis:
        change_config(
            {
                'matching': algo,
                'behaviour': strat,
                'num_taxis': 10000,
                'hard_limit': 35,
                # 'start_date': date[0],
                # 'end_date': date[1]
            }
        )
            
        make_config_files_and_run()

        # if strat == "ml_dispatcher_v2":
        #    change_config({'matching': algo, 'behaviour': strat, 'ml_weighted': True})
        #    make_config_files_and_run()
