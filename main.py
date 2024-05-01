import streamlit as st
import pandas as pd
import os


# Define function to display dataset
def display_dataset():
    st.subheader('Dataset')
    dataset_path = "AESO_2020_demand_price.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        st.write(df)
    else:
        st.write("Dataset file not found.")


# Define function to display additional output for LSTM and Q Learning
def display_additional_output(option):
    if option == 'LSTM':
        st.title('Smart scheduling strategy for charging of the electric vehicle')
        st.subheader('LSTM Code')
        with open('LSTMW.py', 'r') as file:
            code = file.read()
        st.code(code, language='python')

        st.subheader('LSTM Output')
        lstm_output = """
        219/219 [==============================] - 7s 9ms/step - loss: 0.0907 - val_loss: 0.0382
        Epoch 2/200
        219/219 [==============================] - 1s 4ms/step - loss: 0.0164 - val_loss: 0.0271
        Epoch 3/200
        219/219 [==============================] - 1s 6ms/step - loss: 0.0096 - val_loss: 0.0162
        Epoch 4/200
        219/219 [==============================] - 1s 5ms/step - loss: 0.0056 - val_loss: 0.0062
        Epoch 5/200
        219/219 [==============================] - 1s 6ms/step - loss: 0.0031 - val_loss: 0.0018
        Epoch 6/200
        219/219 [==============================] - 1s 6ms/step - loss: 0.0022 - val_loss: 2.2749e-04
        Epoch 7/200
        219/219 [==============================] - 2s 11ms/step - loss: 0.0018 - val_loss: 4.0868e-05
        Epoch 8/200
        219/219 [==============================] - 1s 7ms/step - loss: 0.0016 - val_loss: 6.1423e-05
        Epoch 9/200
        219/219 [==============================] - 1s 6ms/step - loss: 0.0015 - val_loss: 7.4833e-05
        Epoch 10/200
        219/219 [==============================] - 2s 7ms/step - loss: 0.0015 - val_loss: 1.1247e-04
        """
        st.code(lstm_output, language='text')

        st.subheader('Model Summary')
        model_summary = """
        Model: "sequential"
        __________________________________________________________________
         Layer (type)                Output Shape              Param #   
        ==================================================================
         lstm (LSTM)                 (None, 50)                10800     

         dropout (Dropout)           (None, 50)                0         

         dense (Dense)               (None, 1)                 51        

        Total params: 10851 (42.39 KB)
        Trainable params: 10851 (42.39 KB)
        Non-trainable params: 0 (0.00 Byte)
        __________________________________________________________________
        """
        st.code(model_summary, language='text')

        st.subheader('Mean Absolute Percentage Error (MAPE)')
        st.write('0.24184039392357773')

        st.subheader('Demand Prediction')
        st.image('DM.LSTM.jpg')
        st.subheader('Model Loss')
        st.image('ML.LSTM.jpg')

    elif option == 'Comparison':
        st.title('Smart scheduling strategy for charging of the electric vehicle')
        st.subheader('Comparison Code')
        with open('simulation.py', 'r') as file:
            code = file.read()
        st.code(code, language='python')

        st.subheader('Comparison Output')
        q_learning_output = """
        Charging Schedule:
        Hour 17:00 - 18:00: 4 EVs charging
        Hour 20:00 - 21:00: 4 EVs charging
        Hour 21:00 - 22:00: 3 EVs charging
        Hour 22:00 - 23:00: 3 EVs charging
        Hour 1:00 - 2:00: 3 EVs charging
        Hour 18:00 - 19:00: 1 EVs charging
        Hour 19:00 - 20:00: 4 EVs charging
        Hour 3:00 - 4:00: 1 EVs charging
        Hour 6:00 - 7:00: 2 EVs charging
        Hour 23:00 - 24:00: 1 EVs charging
        Hour 4:00 - 5:00: 2 EVs charging
        Hour 7:00 - 8:00: 4 EVs charging
        Hour 0:00 - 1:00: 2 EVs charging
        Hour 5:00 - 6:00: 2 EVs charging

        Discharging Schedule:
        Hour 0:00 - 1:00: 2 EVs discharging
        Hour 17:00 - 18:00: 2 EVs discharging
        Hour 20:00 - 21:00: 3 EVs discharging
        Hour 23:00 - 24:00: 4 EVs discharging
        Hour 2:00 - 3:00: 3 EVs discharging
        Hour 5:00 - 6:00: 5 EVs discharging
        Hour 22:00 - 23:00: 4 EVs discharging
        Hour 3:00 - 4:00: 5 EVs discharging
        Hour 21:00 - 22:00: 4 EVs discharging
        Hour 6:00 - 7:00: 4 EVs discharging
        Hour 18:00 - 19:00: 4 EVs discharging
        Hour 1:00 - 2:00: 2 EVs discharging
        Hour 4:00 - 5:00: 4 EVs discharging
        Hour 7:00 - 8:00: 1 EVs discharging
        """
        st.code(q_learning_output, language='text')
        st.subheader('Comparison Output')
        st.image('QL.jpg')

    elif option == 'Q Learning':
        st.title('Smart scheduling strategy for charging of the electric vehicle')
        st.subheader('Q learning Code')
        with open('QlearningC.py', 'r') as file:
            code = file.read()
        st.code(code, language='python')

        st.subheader('Q learning Output')
        q_learning_output = """
        Experiment parameters are:
        ('n_episodes', 10)
        ('n_hours', 15)
        ('n_divisions_for_soc', 4)
        ('n_divisions_for_percent_honesty', 3)
        ('max_soc_allowed', 1)
        ('min_soc_allowed', 0.1)
        ('alpha', 0.01)
        ('epsilon', 0.1)
        ('gamma', 1)
        ('total_cars_in_alberta', 1000.0)
        ('ev_market_penetration', 0.1)
        ('charging_soc_addition_per_time_unit_per_ev', 0.15)
        ('discharging_soc_reduction_per_time_unit_per_ev', 0.15)
        ('charging_soc_mw_addition_to_demand_per_time_unit_per_ev', 0.01)
        ('discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev', 0.01)
        ('driving_soc_reduction_per_km_per_ev', 0.0035)
        ('alberta_average_demand', array([ 9.07035165,  8.98092308,  8.93776923,  8.9523956 ,  9.02982418,
                9.2411978 ,  9.54387912,  9.79602198,  9.9409011 , 10.019     ,
               10.06354945, 10.06320879, 10.06035165, 10.04946154, 10.05223077,
               10.08514286, 10.13913187, 10.13862637, 10.09064835, 10.0443956 ,
                9.9500989 ,  9.74667033,  9.44698901,  9.19865934,  9.04031868,
                8.95108791,  8.9159011 ,  8.93157143,  9.01947253,  9.24103297,
                9.55554945,  9.81103297,  9.95027473, 10.02832967, 10.07141758,
               10.07118681, 10.06568132, 10.0483956 , 10.05310989, 10.08959341,
               10.14830769, 10.13820879, 10.08635165, 10.03215385,  9.93415385,
                9.7226044 ,  9.42098901,  9.18001099,  9.03238462,  8.94428571,
                8.90665934,  8.91634066,  9.00035165,  9.20993407,  9.52126374,
                9.78102198,  9.93275824, 10.01383516, 10.06843956, 10.078     ,
               10.0759011 , 10.06563736, 10.05894505, 10.09840659, 10.15936264,
               10.14823077, 10.1009011 , 10.05238462,  9.95396703,  9.74908791,
                9.45472527,  9.20893407,  9.0606044 ,  8.97069231,  8.93225275,
                8.9421978 ,  9.02487912,  9.23713187,  9.55318681,  9.81130769,
                9.95765934, 10.03202198, 10.07056044, 10.08149451, 10.08643956,
               10.07441758, 10.07276923, 10.10647253, 10.16142857, 10.15373626,
               10.09926374, 10.05736264,  9.97061538,  9.76169231,  9.46752747,
                9.22062637]))
        ('index_to_time_of_day_dict', {0: 17, 1: 18, 2: 19, 3: 20, 4: 21, 5: 22, 6: 23, 7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 14: 7})
        ('forecast_flag', False)
        ('n_percent_honesty', ['0.25', '0.5', '0.75'])
        ('which_avg_param', 1)
          0%|                                                                        | 0/10 [00:00<?, ?it/s] """
        st.code(q_learning_output, language='text')
        st.subheader('Simulation Summary')
        simulation_summary = """
        
        Run name:  test.py
        Last max load:  10.298626373626373
        Last average:  9.709225022893772
        Reward:  -2.0607052930942302
        PAR:  1.0607052930942302
        
        
        Run name:  test.py
        Last max load:  10.258626373626372
        Last average:  9.711308356227105
        Reward:  -2.056358834188219
        PAR:  1.0563588341882189
        
        
        Run name:  test.py
        Last max load:  10.338626373626372
        Last average:  9.758391689560439
        Reward:  -2.0594600731887684
        PAR:  1.0594600731887684
        
        
        Run name:  test.py
        Last max load:  10.668626373626372
        Last average:  9.759641689560441
        Reward:  -2.093137095907757
        PAR:  1.0931370959077567
        
        
        Run name:  test.py
        Last max load:  10.260648351648351
        Last average:  9.682975022893773
        Reward:  -2.05965866145361
        PAR:  1.0596586614536097
        
        
        Run name:  test.py
        Last max load:  10.590648351648351
        Last average:  9.696725022893775
        Reward:  -2.092188169371003
        PAR:  1.0921881693710032
        
        
        Run name:  test.py
        Last max load:  10.28064835164835
        Last average:  9.6850583562271
        Reward:  -2.061495757022291
        PAR:  1.061495757022291
        
        
        Run name:  test.py
        Last max load:  10.4900989010989
        Last average:  9.7138083562271
        Reward:  -2.079916189037655
        PAR:  1.0799161890376552
        
        
        Run name:  test.py
        Last max load:  10.476021978021977
        Last average:  9.750475022893772
        Reward:  -2.07441144697306
        PAR:  1.07441144697306
        
        
        Run name:  test.py
        Last max load:  10.794395604395604
        Last average:  9.708808356227104
        Reward:  -2.1118146747095095
        PAR:  1.1118146747095092
        100%|███████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 91.02it/s]
                        """
        st.code(simulation_summary, language='text')
        st.subheader('Q Learning Output')
        st.image('QL.jpg')


def main():
    st.sidebar.title('Options')
    option = st.sidebar.selectbox('', ['Data', 'LSTM', 'Q Learning', 'Comparison', 'Report'])

    if option == 'Data':
        display_dataset()

    elif option == 'LSTM':
        st.title('LSTM Option')
        display_additional_output(option)

    elif option == 'Comparison':
        st.title('Comparison Option')
        display_additional_output(option)

    elif option == 'Q Learning':
        st.title('Q Learning Option')
        display_additional_output(option)

    elif option == 'Report':
        st.title('Report Option')
        # Add your report content here
        st.write('This is the report content.')


if __name__ == "__main__":
    main()
