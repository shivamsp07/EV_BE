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

        # Adding report content
        st.header('A Smart Charging and Discharging Scheduling Strategy for the Electric Vehicle')
        st.write('**Authors:** Dr. Archana Y. Chaudhari, Ved Inamdar, Shivam Pawar, Devansh Kariya, Rajat Parate.')
        st.write('**Affiliation:** Dept. of Information Technology, Dr. D. Y. Patil Institute of Technology, Pimpri, Pune Maharashtra, India.')
    
        st.subheader('Abstract')
        st.write("One of the most important steps toward a low-carbon economy and sustainable energy future is the switch to electric vehicles (EVs). However, because of their high charging requirements, the broad adoption of EVs presents a risk to the reliability of the electrical system. Strategies for scheduling charging and discharging that work are essential to reducing the negative grid effects of EVs. This paper explores two strategies for intelligent charging and discharging scheduling: Q-learning and Long Short-Term Memory (LSTM) coupled with Integer Linear Programming (ILP). The LSTM-ILP approach uses ILP to optimize the charging and discharging schedules while utilizing deep learning to anticipate energy consumption. On the other hand, the Q-learning method makes use of reinforcement learning to ascertain the best course of action for EVs in relation to their state-of-charge (SOC) and the demand on the grid. The outcomes of the simulation show that both strategies are successful in lowering the peak-to-average ratio of the grid and lessening the influence of EV charging demands. Comparative assessments draw attention to the advantages and disadvantages of each approach, offering suggestions for further study and real-world application. Key words: Q-Learning, LSTM-ILP, Grid")

        
    
        st.subheader('1. Introduction')
        st.write('The transition towards electric vehicles (EVs) is a critical part of the shift to a low-carbon economy and sustainable energy future. Governments and international bodies like the International Energy Agency (IEA) have set ambitious targets, estimating 30% of all vehicles to be electric by 2030. These new electric vehicles will need to be charged, mostly around 5-6 PM, which creates a high load spike on the grid. At 10% market penetration of EVs, an increase in peak load demand is estimated to be around 18%, and much higher for higher levels of market penetration. However, the ability of EV batteries to discharge energy back to the grid provides benefits like peak load shaving, frequency regulation, spinning reserves, and improved grid stability and efficiency.')
        st.write('The concept of vehicle-to-grid (V2G), first proposed by Amory Lovins in 1995, aims to harness the bidirectional energy flow capabilities of EVs to improve grid operations through smart charging and discharging control strategies. Currently, price-based demand response schemes like real-time pricing, time-of-use tariffs and dynamic pricing are the main methods explored to incentivize EVs to participate in V2G. However, these methods face challenges like guaranteeing overall system optimality while respecting EV flexibility constraints, accounting for EV randomness and uncertainty, avoiding excessive grid fluctuations due to frequent price changes, and ensuring scalability as EV numbers increase.')
        st.write('This work aims to develop a novel smart charging/discharging scheduling strategy for EVs participating in V2G that addresses the above limitations. By combining Long Short-Term Memory (LSTM) with Integer Linear Programming (ILP), and Q-learning, the proposed strategy aims to reduce peak-to-average ratio (PAR) of grid load through effective peak shaving and valley filling, minimizing EV charging costs for users while respecting their mobility needs, and ensuring grid-EV coordination scalability - all while maintaining system robustness to uncertainties and increasing EV penetration levels.')
    
        st.subheader('2. Related Work')
        st.write('In recent years, researchers have been actively exploring innovative strategies to optimize EV charging schedules. Several methodologies and their associated advantages and limitations are reviewed. For example, Yanyu Zhang and colleagues (2023) presented a cooperative EV charging scheduling strategy based on double deep Q-network and prioritized experience replay. Their approach utilizes deep reinforcement learning to address the EV charging scheduling problem, achieving collaborative scheduling among EVs and reducing charging costs. However, it falls short in considering the coordination of renewable energy systems and transformer loads.')
        st.write('Other notable contributions include Shuai Li and co-authors who proposed a Distributed Transformer Joint Optimization Method using Multi-Agent Deep Reinforcement Learning for EV Charging. This approach focuses on coordinating EV charging while safeguarding user privacy and reducing communication equipment deployment costs. Nevertheless, it neglects renewable energy integration and the challenge of reward sparsity in learning scenarios.')
        st.write('Another example is the work of Lina Ren and Mingming Yuan (2023), briefly mentioned but lacking details on their electric vehicle charging and discharging scheduling strategy.')
    
        st.subheader('3. System Model')
        st.write('The proposed plan is predicated on several assumptions: all EVs are the same, there is always enough capacity to supply any amount of power into the V2G service, the base energy demand profile is the same for every day of the year but varies hourly, and the presence of EVs does not change Alberta\'s base load. Fig. 1 depicts the demand load for Alberta. Every hour, either 25%, 50%, or 75% of EVs take part. Every hour, the degree of involvement is known at the outset. At 5 PM, when every EV\'s SOC reaches 30% on average with a 10% standard deviation, the V2G service begins. EVs can charge, discharge, or do nothing between 5 PM and 3 AM as long as they use the V2G service. Seventy-five percent of EVs use the V2G service starting at 3 AM. At 8 AM, the V2G service terminates.')
        st.image('F1.png', caption='Fig. 1. Visualization of the data set used')
    
        st.subheader('3.1 LSTM-ILP')
        st.write('LSTM is a type of recurrent neural network (RNN) designed to address issues like gradient explosion or disappearance commonly found in traditional RNNs. It features interconnected neuron layers with memory cells, capable of retaining information from previous time steps and transmitting it forward, making it suitable for time-related tasks.')
        st.write('**LSTM Design:** Two hidden layers with 36 nodes each. Inputs: Grid base load (fbase), collective EV charging demand load (fcharging), discharge available load (fdischarge), and large grid electricity price (R). Outputs: 24-hour electricity price (r).')
        st.image('F2.png', caption='Fig. 2. LSTM-ILP Model')
    
        st.subheader('3.2 Linear Programming (LP)')
        st.write('The goal of linear programming (LP) is to maximize choices within linear constraints. Typically, this process consists of three stages: issue analysis, objective function establishment, and variable limitation determination.')
        st.write('**Optimization Process:**')
        st.write('- Decision Variables: Charging and discharging power.')
        st.write('- Objective Function: Minimize peak-to-valley grid load difference and EV charging and discharging costs.')
        st.write('- Constraints: Ensure safety and technical feasibility.')
    
        st.subheader('3.3 Improved Linear Programming (ILP)')
        st.write('ILP enhances LP by subsidizing electricity prices for EVs participating in V2G, aiming to address rapid grid load changes more effectively.')
        st.write('**Improvement Process:**')
        st.write('- Redistribute chargeable and dischargeable loads if the absolute difference between charging and discharging electricity prices is less than a threshold.')
        st.write('- Incorporate a new constraint to ensure load redistribution.')
        st.write('- Return loss incurred by EV owners as a subsidy, also fed back to LSTM as part of its loss function.')
    
        st.subheader('3.4 Q-Learning')
        st.write('A Markov Decision Process (MDP) temporal-difference learning issue is suggested to be derived from a multi-objective, multi-agent cooperative game minimization problem. Traditionally, reinforcement learning involves an agent interacting with its surroundings and choosing behaviors in states that will result in rewards, with the goal of maximizing the total of all future rewards.')
        st.image('F3.png', caption='Fig. 3. Traditional Q-learning')
        st.write('This figure illustrates the classic Q-learning process, where the agent engages with the environment by acting in various states and gaining rewards. By updating the Q-values, or projected future rewards, for each state-action combination, the objective is to develop an optimal policy that maximizes the cumulative reward over time.')
    
        st.write('The proposed algorithm initializes Q-values for all state-action pairs and iteratively updates them for each episode. Actions are chosen using an ε-greedy strategy to balance exploration and exploitation. Rewards and next states are observed, and Q-values are updated accordingly. The algorithm continues until the end of the V2G service.')
    
        st.image('F4.png', caption='Fig. 4. Multi-Agent Q-Learning')
        st.write('This figure illustrates the multi-agent cooperative reinforcement learning configuration utilized in the suggested Q-learning method. In this scenario, a number of agents (EVs) work toward the same goal (keeping SOC and minimizing peak-to-average ratio) and are rewarded for it.')

        


if __name__ == "__main__":
    main()
