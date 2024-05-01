# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import argparse
import os

def get_input_args():
    """
    Returns input arguments for main file execution
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10,
                        help='Number of episodes to run')
    parser.add_argument('--id_run', type=str, default='test_run',
                        help='id of run')
    parser.add_argument('--pen', type=float, default=0.1,
                        help='market penetration of evs')
    parser.add_argument('--avg_param', type=int, default=1,
                        help='if avg == 1, non-one avg and non-zero max are used')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='alpha for learning')
    parser.add_argument('--scale', type=int, default=1,
                        help='scale')
    return parser.parse_args()

# Get args
# n_episodes = get_input_args().n
# id_run = get_input_args().id_run
# pen = get_input_args().pen
# avg = get_input_args().avg_param
# alpha = get_input_args().alpha
# scale = get_input_args().scale
n_episodes = 10
id_run = "test.py"
pen = 0.1
avg = get_input_args().avg_param
alpha = 0.01
scale = 1000

# Get Alberta Average demand and prices
df = pd.read_csv('AESO_2020_demand_price.csv')
HE = []
end_index = df.shape[0]//(48 * 2) + 1
for day in range(1, end_index):
    for hour in range(1, (2 * 48) + 1):
        HE.append(hour)
df['HE'] = HE
df = df.drop(df.columns[[0, 2]], axis=1)
df = df.set_index('HE', drop=True)
df = df.groupby('HE', as_index=True).mean()
df_to_plot = df.drop(df.columns[[0]], axis=1)

alberta_avg_power_price = np.array(df.iloc[:, 0])
alberta_avg_demand = np.array(df.iloc[:, 1])/scale

# https://open.alberta.ca/dataset/d6205817-b04b-4360-8bb0-79eaaecb9df9/
# resource/4a06c219-03d1-4027-9c1f-a383629ab3bc/download/trans-motorized-
# vehicle-registrations-select-municipalities-2020.pdf
total_cars_in_alberta = 100
ev_market_penetration = 0.1
min_soc_by_8_am = 0.5
max_soc_allowed = 1
min_soc_allowed = 0.1
charging_soc_addition_per_time_unit_per_ev = 0.15
discharging_soc_reduction_per_time_unit_per_ev = -0.15
charging_soc_mw_addition_to_demand_per_time_unit_per_ev = 0.01
discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev = 0.01
driving_soc_reduction_per_time_unit_per_ev = 0.005
forecast_flag = False
n_percent_honesty = ['0.25', '0.5', '0.75']

# Time conversion
index_of_time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
time_of_day = [17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]

index_to_time_of_day_dict = {}
for item in range(len(index_of_time)):
    index_to_time_of_day_dict[index_of_time[item]] = time_of_day[item]
pprint(index_to_time_of_day_dict)

# Define experiment params
experiment_params = {'n_episodes': n_episodes,
                     'n_hours': 15,
                     'n_divisions_for_soc': 4,
                     'n_divisions_for_percent_honesty': 3,
                     'max_soc_allowed': 1,
                     'min_soc_allowed': 0.1,
                     'alpha': alpha,
                     'epsilon': 0.1,
                     'gamma': 1,
                     'total_cars_in_alberta': 1000000/scale,
                     'ev_market_penetration': pen,
                     'charging_soc_addition_per_time_unit_per_ev': 0.15,
                     'discharging_soc_reduction_per_time_unit_per_ev': 0.15,
                     'charging_soc_mw_addition_to_demand_per_time_unit_per_ev': 0.01,
                     'discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev': 0.01,
                     'driving_soc_reduction_per_km_per_ev': 0.0035,
                     'alberta_average_demand': alberta_avg_demand,
                     'index_to_time_of_day_dict': index_to_time_of_day_dict,
                     'forecast_flag': forecast_flag,
                     'n_percent_honesty': n_percent_honesty,
                     'which_avg_param': avg
                     }


# Experiment function
class Experiment():
    def __init__(self, params):
        self.params = params

    def add_to_dict(self, action, hour, dict_):
        if hour in dict_:
            dict_[hour].append(action)
        else:
            dict_[hour] = [action]
        return dict_

    def run_experiment(self):
        charge_history = {}
        discharge_history = {}
        for i in range(self.params['n_episodes']):
            state = np.random.uniform(self.params['min_soc_allowed'],
                                      self.params['max_soc_allowed'])
            for j in range(self.params['n_hours']):
                # Implement epsilon-greedy policy here
                if np.random.uniform(0, 1) < self.params['epsilon']:
                    action = np.random.choice([0, 1, 2])
                else:
                    action = np.argmax(np.random.random(3))

                # Ensure not charging if SOC is already at max
                if action == 1 and state == self.params['max_soc_allowed']:
                    action = 0

                # Ensure not discharging if SOC is already at min
                if action == 2 and state == self.params['min_soc_allowed']:
                    action = 0

                # Update state and charge/discharge history
                if action == 0:  # Do nothing
                    pass
                elif action == 1:  # Charge
                    state = min(state + self.params['charging_soc_addition_per_time_unit_per_ev'],
                                self.params['max_soc_allowed'])
                    charge_history = self.add_to_dict(action, j, charge_history)
                elif action == 2:  # Discharge
                    state = max(state + self.params['discharging_soc_reduction_per_time_unit_per_ev'],
                                self.params['min_soc_allowed'])
                    discharge_history = self.add_to_dict(action, j, discharge_history)

            # Update Q-values - not implemented for simplicity
            # self.Q[state, action] = ...

        return charge_history, discharge_history


# Run the experiment
experiment = Experiment(experiment_params)
charge_history, discharge_history = experiment.run_experiment()

# Print the charging and discharging schedule
print("Charging Schedule:")
for hour, actions in charge_history.items():
    print(f"Hour {experiment_params['index_to_time_of_day_dict'][hour]}:00 - {experiment_params['index_to_time_of_day_dict'][hour]+1}:00: {len(actions)} EVs charging")

print("\nDischarging Schedule:")
for hour, actions in discharge_history.items():
    print(f"Hour {experiment_params['index_to_time_of_day_dict'][hour]}:00 - {experiment_params['index_to_time_of_day_dict'][hour]+1}:00: {len(actions)} EVs discharging")

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))  # Increase the width of the figure to accommodate all labels

charge_hours = [hour for hour, actions in charge_history.items() if actions]
discharge_hours = [hour for hour, actions in discharge_history.items() if actions]

for hour, actions in charge_history.items():
    ax.bar(hour, len(actions), color='b', alpha=0.5)

for hour, actions in discharge_history.items():
    ax.bar(hour, -len(actions), color='r', alpha=0.5)

ax.set_xlabel('Hour of the day')
ax.set_ylabel('Number of EVs')
ax.set_title('Charging and Discharging Schedule')
ax.set_xticks(list(charge_history.keys()))
ax.set_xticklabels([str(hour) + ":00" for hour in charge_history.keys()], rotation=45, ha='right')

# Adding legend with different colors
legend = ax.legend(['Charging', 'Discharging'], loc='upper right')
legend.legendHandles[0].set_color('blue')
legend.legendHandles[1].set_color('red')

# Customize color of x-axis labels
ax.tick_params(axis='x', colors='black')

plt.tight_layout()  # Adjust layout to prevent clipping of x-axis labels
plt.show()
