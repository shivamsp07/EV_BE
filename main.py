import streamlit as st
import pandas as pd
import os


# Define function to display dataset
def display_dataset():
    st.subheader('Dataset')
    st.title('Smart scheduling strategy for charging and discharging of the electric vehicle')
    dataset_path = "AESO_2020_demand_price.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        st.write(df)
    else:
        st.write("Dataset file not found.")


# Define function to display additional output for LSTM and Q Learning
def display_additional_output(option):
    if option == 'LSTM':
        st.title('Smart scheduling strategy for charging and discharging of the electric vehicle')
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
        st.image('Q.OT.jpg')


def main():
    st.sidebar.title('Options')
    option = st.sidebar.selectbox('', ['Data', 'LSTM', 'Q Learning', 'Report'])

    if option == 'Data':
        display_dataset()

    elif option == 'LSTM':
        st.title('LSTM Option')
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
        st.write("The transition towards electric vehicles (EVs) is a critical part of the shift to a low-carbon economy and sustainable energy future. Governments and international bodies like the International Energy Agency (IEA) have set ambitious targets, estimating 30All these new electric vehicles will need to be charged. Right now, most electric vehicles begin charging around 5-6 PM which creates a high load spike on the grid. In the case of 10% market penetration of EVs an increase in the peak load demand is estimated to be around 18%, and much higher for higher levels of market penetration. On one hand, the ability of EV batteries to discharge energy back to the grid provides substantial benefits like peak load shaving, frequency regulation, spinning reserves, and improved grid stability and efficiency [5,6] But there is also a chance presented by this difficulty. Through the use of vehicle-to-grid (V2G) technology, EVs may function as mobile energy storage devices, storing excess renewable energy and returning it to the grid during periods of high demand. The power system's resilience and sustainability may be improved by the combination of EVs with renewable energy sources.")

        st.write("However, if uncontrolled, the high coincidence of EV charging times with existing residential peak loads can exacerbate grid stress, voltage fluctuations, transformer overloads and a dramatic increase in peak demand [3,4,7]. The concept of vehicle-to-grid (V2G), first proposed by Amory Lovins in 1995 [Tomi and Kempton, 2007; Guille and Gross, 2009; Shukla, 2018], aims to harness the bidirectional energy flow capabilities of EVs to improve grid operations through smart charging and discharging control strategies. Currently, price-based demand response schemes like real-time pricing, time-of-use tariffs and dynamic pricing are the main methods explored to incentivize EVs to participate in V2G [Aljohani et al., 2021; Zhou et al., 2021; Yu et al., 2020; Zhong-Fu, 2008; Aghajani et al., 2017; Sadati et al., 2018]. Several recent works have investigated dynamic pricing approaches for V2G using techniques like stochastic programming [Luo et al., 2018], Bayesian prediction models [Dante et al., 2022], Markov decision processes [Fang et al., 2021] and reinforcement learning [Liu et al., 2021].")

        st.write("However, these methods face challenges in aspects like guaranteeing overall system optimality while respecting EV flexibility constraints, accounting for EV randomness and uncertainty, avoiding excessive grid fluctuations due to frequent price changes, and ensuring scalability as EV numbers increase. This work aims to develop a novel smart charging/discharging scheduling strategy for EVs participating in V2G that addresses the above limitations. By combining Long Short-Term Memory (LSTM) combined with Integer Linear Programming (ILP), and Q-learning, the proposed strategy is designed to achieve objectives like reducing peak-to-average ratio (PAR) of grid load through effective peak shaving and valley filling, minimizing EV charging costs for users while respecting their mobility needs, and ensuring grid-EV coordination scalability - all while maintaining system robustness to uncertainties and increasing EV penetration levels.")

        st.subheader('2. Related Work')
        st.write("In recent years, researchers have been actively exploring innovative strategies to optimize Electric Vehicle (EV) charging schedules. A comprehensive review of several research papers reveals various methodologies and their associated advantages and limitations. Yanyu Zhang and colleagues, in their paper published in 2023, present a cooperative EV charging scheduling strategy based on double deep Q-network and prioritized experience replay. Their approach utilizes deep reinforcement learning to address the EV charging scheduling problem and effectively achieve collaborative scheduling among EVs, ultimately reducing charging costs and promoting PV energy consumption. However, it falls short in considering the coordination of renewable energy systems and transformer loads [1].")

        st.write("Another noteworthy contribution comes from Shuai Li and co-authors, who proposed a Distributed Transformer Joint Optimization Method using Multi-Agent Deep Reinforcement Learning for EV Charging. This approach focuses on coordinating EV charging while safeguarding user privacy and reducing communication equipment deployment costs. Nevertheless, it neglects renewable energy integration and the challenge of reward sparsity in learning scenarios [8]. In contrast, Lina Ren and Mingming Yuan’s work in 2023 is briefly mentioned, but the details of their electric vehicle charging and discharging scheduling strategy are absent from the table [2].")
        
        st.write("Moving to the year 2021, the IEEE Internet of Things Journal features a paper introducing CDDPG, a Deep Reinforcement Learning-Based Approach for Electric Vehicle Charging Control. This approach offers the advantage of considering multiple objectives simultaneously and addresses the reward sparsity issue with two replay buffers. However, similar to the previous works, it does not take into account the coordination of renewable energy and transformer load, and its simulation results are location-specific [5].")
        
        st.write("Furthermore, a study published in 2021 by F. Tuchnitz and his team in the Applied Energy journal presents a smart charging strategy for an electric vehicle fleet based on reinforcement learning. This strategy optimizes EV charging in a fleet without prior knowledge of system dynamics, offering a flexible and scalable approach that significantly reduces the variance of the total load. Nevertheless, the simulation results are limited to one specific case study, potentially limiting the generalizability of the findings [9].")
        
        st.write("In a paper published in 2020 in IEEE Transactions on Smart Grid, a multi-agent reinforcement learning approach is utilized to coordinate electric vehicle charging. This method can handle various tariff types and consumer preferences while minimizing assumptions about the distribution grid. Despite the success in balancing energy costs and transformer load, the study is constrained to experiments with real energy prices and may not be universally applicable [10].")
        
        st.write("Lastly, in 2018, Xu Hao and Yue Chen presented an A V2G-oriented reinforcement learning framework for heterogeneous electric vehicle charging management. This approach employs deep Q-network reinforcement learning to optimize EV charging in Vehicle-to-Grid (V2G) systems and accounts for uncertainties and EV heterogeneity, leading to significant cost reductions compared to traditional methods. Nevertheless, the study is limited to Chinese cities and may not be applicable to other regions. Additionally, it assumes the availability of departure-time information and does not consider the coordination of renewable energy and transformer load, along with potential overestimation issues [11].")
        
        st.write("While these papers provide valuable insights into EV charging scheduling strategies, it is essential to consider their specific limitations and address them in future research to develop comprehensive and adaptable solutions for efficient EV charging.")

        st.subheader('3. System Model')
        st.write('The proposed plan is predicated on several assumptions: all EVs are the same, there is always enough capacity to supply any amount of power into the V2G service, the base energy demand profile is the same for every day of the year but varies hourly, and the presence of EVs does not change Alberta\'s base load. Fig. 1 depicts the demand load for Alberta. Every hour, either 25%, 50%, or 75% of EVs take part. Every hour, the degree of involvement is known at the outset. At 5 PM, when every EV\'s SOC reaches 30% on average with a 10% standard deviation, the V2G service begins. EVs can charge, discharge, or do nothing between 5 PM and 3 AM as long as they use the V2G service. Seventy-five percent of EVs use the V2G service starting at 3 AM. At 8 AM, the V2G service terminates.')
        st.image('F1.png', caption='Fig. 1. Visualization of the data set used')
        st.write("The above figure (Fig. 1.) illustrates the load profile or demand pattern employed in the. It shows how the total EV demand load (the entire amount of energy that EVs can demand to be charged), the total EV discharge capacity (the total amount of energy that EVs can potentially discharge into the grid), and the grid foundation load (the baseline load without EVs) change over time. The input data for the suggested scheduling technique is this load profile.")
    
        st.subheader('3.1 LSTM-ILP')
        st.write('LSTM is a type of recurrent neural network (RNN) designed to address issues like gradient explosion or disappearance commonly found in traditional RNNs. It features interconnected neuron layers with memory cells, capable of retaining information from previous time steps and transmitting it forward, making it suitable for time-related tasks.')
        st.write("Ft = σ(WfHt − 1 + Pfxt + df) (1)")
        st.write("It = σ(W ∗ iH ∗ t − 1 + Pixt + di) (2)")
        st.write("Cet = Tanh(W ∗ cH ∗ t − 1 + Pcxt + dc) (3)")
        st.write("Ct = F ∗ t ⊙ C ∗ t − 1 + It ⊙ Cet (4)")
        st.write("Ot = σ(W ∗ oH ∗ t − 1 + Poxt + do) (5)")
        st.write("Ht = Ot ⊙Tanh(Ct) (6)")
        st.write("t= σ(VHt + dy) (7)")
        st.write('**LSTM Design:** Two hidden layers with 36 nodes each. Inputs: Grid base load (fbase), collective EV charging demand load (fcharging), discharge available load (fdischarge), and large grid electricity price (R). Outputs: 24-hour electricity price (r).')
        st.image('F2.png', caption='Fig. 2. LSTM-ILP Model')
        st.write("The above figure (Fig. 2.) illustrates the architecture of the study's suggested LSTM-ILP (Long Short-Term Memory - Integer Linear Programming) model. The LSTM neural network, which is its main component, predicts dynamic energy pricing by considering input parameters such as EV demand, discharge capacity, and grid foundation load. The updated linear programming optimization model considers the predicted prices and uses them to determine the best EV charging and discharging schedules while considering variables like load difference minimization, maintaining a sufficient EV battery state-of-charge (SOC), and adding subsidies for EV owners taking part in the vehicle-to-grid (V2G) program.")

    
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

        st.write("Agents in multi-agent cooperative reinforcement learning have the same goal and are rewarded for each transition. Goals O1 and O2 are linearly weighted with weights w1 and w2, resulting in the creation of a single goal, A time slot (h) and a participation percentage (Y_h) are combined to establish each state. Random selections of charging, discharging, or doing nothing at each time interval comprise actions A_h. Agents are incentivized by the penalty P_h to charge EVs such that, by the time the V2G service expires, the mean SOC meets a minimal level. Each time slot's reward, r_h, is determined by combining the penalty for SOC deviations with the negative peak-to-average ratio (PAR) in a linear fashion.")

        st.write("Algorithm 1 Proposed Algorithm for Long Short-Term Memory - Improved Linear Programming (LSTM-ILP)")
        st.write("i : Initialize parameters:")
        st.write("ii : - Grid base load (fbase)")
        st.write("iii : - Collective EV charging demand load (fcharging)")
        st.write("iv : - Discharge available load (fdischarge)")
        st.write("v : - Large grid electricity price (R)")
        st.write("vi : - Threshold for absolute difference between charging and discharging electricity prices (δr)")
        st.write("vii : - Subsidy parameter (η)")
        st.write("viii : Define LSTM neural network architecture:")
        st.write("ix : - Input layer: fbase,fcharging,fdischarge,R")
        st.write("x : - Two hidden layers with 36 nodes each")
        st.write("xi : - Output layer: 24-hour electricity price (r)")
        st.write("xii : Train LSTM network using historical data.")
        st.write("xiii : Linear Programming (LP):")
        st.write("0: a. Define decision variables:")
        st.write("- Charging power (Pc) 0: - Discharging power (Pd)")
        st.write("b. Establish objective function:")
        st.write("- Minimize peak-to-valley grid load difference (δf) and EV charging and discharging costs (Ri)")
        st.write("Objective Function:")
        st.write("minδf,Ri")
        st.write("c. Set constraints:")
        st.write("- Safety and technical constraints on charging and discharging power (Pc,Pd)")
        st.write("- Battery state of charge constraints (SOC)")
        st.write("- Charge balance constraints")
        st.write("- Grid load constraints")
        st.write("i. : Improved Linear Programming (ILP):")
        st.write("ii. : a. Determine if absolute difference between charging and discharging electricity prices is less than δr.")
        st.write("iii. : b. If difference is less than δr:")
        st.write("- Redistribute chargeable and dischargeable loads at those times.")
        st.write("c. Add a new constraint to LP objective function:")
        st.write("- Constraint: Sum of charging and discharging power (Px,Py,...) ≤ average load value (Pavg).")
        st.write("d. Calculate loss function:")
        st.write("Loss 0: Where R1 = P of subsidies for EV owners participating in load balancing.")
        st.write("Return optimized scheduling strategy for each EV, balancing grid load effectively while minimizing costs.")
        
        st.write("Whether the mean SOC can reach a minimum required level by the end of vehicle-to-grid (V2G) service. The penalty encourages agents to charge EVs such that the mean SOC is at least a minimum threshold by the end of V2G service.")
        st.write("The reward rh for each time slot is a linear combination of the negative peak-to-average ratio (PAR) and the penalty for SOC deviations. To solve this problem, the Q-Learning algorithm is utilized. Q-Learning is an off-policy, model-free learning algorithm that updates Q-values for state-action pairs based on observed rewards and next states. The algorithm iteratively learns the optimal policy by updating Q-values towards maximizing future expected rewards.")
        st.write("The proposed algorithm initializes Q-values for all state action pairs and iteratively updates them for each episode. Actions are chosen using an greedy strategy to balance exploration and exploitation. Rewards and next states are observed, and Q-values are updated accordingly. The algorithm continues until the end of the V2G service.")
        st.write("In summary, the proposed algorithm aims to optimize EV charging and discharging schedules by transforming the original problem into an MDP temporal-difference learning problem and utilizing Q-Learning to learn the optimal policy for each state. This approach facilitates efficient and adaptive decision-making to minimize the peak-to-average ratio while maintaining EV state-of-charge within specified constraints.")

        
        st.image('F4.png', caption='Fig. 4. Multi-Agent Q-Learning')
        st.write('This figure illustrates the multi-agent cooperative reinforcement learning configuration utilized in the suggested Q-learning method. In this scenario, a number of agents (EVs) work toward the same goal (keeping SOC and minimizing peak-to-average ratio) and are rewarded for it.')
        st.write("The Q-Learning algorithm (see Algorithm 2) is used to tackle this problem. Q-Learning is a model-free, off-policy learning technique that modifies Q-values for state-action pairings according to the subsequent states and observed rewards. By adjusting Q-values in an iterative manner to maximize future predicted rewards, the algorithm discovers the best course of action. All state-action pairings have their Q-values initialized by the suggested algorithm (Algorithm 3), which then iteratively updates them for every episode. To balance exploration and exploitation, actions are selected using an ε-greedy method. As rewards and subsequent states are detected, Q-values are modified appropriately. The algorithm keeps running until the V2G service expires. To put it briefly, the suggested method converts the original issue into an MDP temporal-difference learning problem and uses Q-Learning to determine the best course of action for each state in order to optimize EV charging and discharging schedules. This strategy makes it easier to make effective, flexible decisions that decrease the peak-to-average ratio while keeping the EV state-of-charge within predetermined bounds.")

        st.subheader('4. Experiments and Results')
        st.subheader('4.1 Q Learning Results')
        st.write("In an effort to maximize vehicle-to-grid (V2G) performance in a residential context with 600 families and 400 EVs, charging and discharging schedules were produced using a Q-learning model.")
        st.write("The charging schedule of the Q-learning model showed that 10 EVs were actively charged between 17:00 and 08:00 in the nighttime and early morning hours, with no discharging activity noted. The objective of this strategy was to optimize grid support capacity during times of lower demand and profit from off-peak power tariffs.")
        st.image('F9.png')

        algorithm_description = """
        0: Input: Learning rate α ∈ [0,1] 
        0: Input: Exploration parameter ϵ < 0
        
        0: Initialization:
        Learning rate α ∈ [0, 1]
        ϵ< 0
        Q(s,a ) s,a Q( , )=0
        S
        0: Initialize Q[(h,Yh),a], for all h,Yh,a,
        Q[(terminal, )] = 0
        
        0: For each episode:
        
        0: for h ∈ [1,2,3,...,NH] do
            Choose Yh+1 at random
            for each SOC bin [1,2,3,...,NSOC bins] do
                Choose action A using ϵ-greedy
                Take chosen action A
                Calculate eh,total except
                Observe reward rh and next state (h + 1,Yh+1)
                for each SOC bin [1,2,3,...,NSOC bins] do
                    Q[(h,Yh),A] ← Q[(h,Yh),A] + α(rh+γ maxa′(Q[(h + 1,Yh+1),a′]) − Q[(h,Yh),A])
                end for
                (h,Yh) ← (h + 1,Yh+1)
            end for
        end for
        """
        
        st.write(algorithm_description)
        st.image('F5.png', caption='Fig. 5. Average Grid Demand per Episode for Q-learning')
        st.write("This figure (Fig. 5.) shows the average grid demand per episode during the training process of the Q-learning algorithm. It illustrates how the grid demand converges as the algorithm learns the optimal policy for EV charging and discharging schedules.")

        st.write("Such meticulous scheduling endeavors underscore the pivotal role of advanced machine learning techniques in orchestrating efficient V2G operations within residential communities. By harnessing the predictive capabilities of LSTM models and the adaptive decision-making prowess of DQN frameworks, aggregators can fine-tune charging and discharging strategies to harmonize with dynamic grid conditions and user preferences. The overarching objective remains twofold: to minimize the burden on the grid during peak hours, thus averting potential strain and enhancing overall reliability, while concurrently optimizing charging patterns to align with cost-effective electricity tariffs and user convenience.")
        
        st.write("The advantages of the proposed method are that EVs can act autonomously knowing only the participation level. Also, the load standard deviation for the full day was generally reduced. However, the method only works on constant load demand data, the SOC of EVs at 8 AM is a mean value, and on any individual day, the mean may be less. Additionally, it is unclear whether the resulting policy is optimal, and some discharging in consecutive hours is present for EVs with less than 25% charge.")
        
        st.write("We proposed a Q-learning-based algorithm for EV demand response in a cooperative multi-agent multi-objective game. While it didn’t reduce PAR, it achieved a 2.7% reduction in average standard deviation of demand load for the full day and a 16.2% reduction for the first 23 hours. After 100,000 epochs, convergence was observed across 30 runs. The method’s advantages include reducing load demand standard deviation and enabling EVs to act independently based only on current participation levels. However, it may discharge EVs when their state of charge is low and doesn’t guarantee a minimum charge in the morning.")

        st.image('F6.png', caption='Fig. 6. Peak-To-Average ratio Per Episode for Q-learning.')

        st.write("This figure (Fig. 6.) illustrates the peak-to-average ratio (PAR) per episode during the training of the Q-learning algorithm. It demonstrates how the algorithm aims to minimize the PAR, which is one of the objectives of the scheduling strategy, by learning the optimal actions for EVs based on their state and participation levels.")

        st.write("The suggested Q-learning approach has two benefits: the load standard deviation for the entire day was largely decreased, and EVs may operate independently with just the participation level known. However, the method's drawbacks include the fact that it can only be applied to continuous load demand data, that the SOC of EVs at 8 AM is a mean value that can vary on any given day, that it's unclear whether the resulting policy is optimal, and that EVs with a charge level of less than 25% will occasionally discharge in consecutive hours.")

        st.subheader('LSTM Results')
        st.write("An LSTM-ILP (Long Short-Term Memory - Integer Linear Programming) model was used to improve the V2G processes in addition to the Q-learning methodology. The charging schedule of the LSTM model demonstrated a purposeful approach of not charging any EVs between 17:00 and 08:00 in the evening, while its discharging schedule showed that 10 EVs were consistently discharged at the same time. This strategy is in line with using V2G capabilities to reduce grid load during periods of high demand, which maximizes energy efficiency and economy.")
        st.image('F7.png', caption='Fig. 7. Model Loss in LSTM-ILP')

        st.write("This figure (Fig. 7.) illustrates the training loss of the LSTM-ILP model over time. The loss function likely combines factors such as the load difference (peak-to-average ratio), EV charging costs, and subsidies provided to EV owners participating in the V2G scheme. Monitoring the loss during training helps assess the model's performance and convergence.")

        st.write("The advantages of the proposed method are that EVs can act autonomously knowing only the participation level. Also, the load standard deviation for the full day was generally reduced. However, the disadvantages are that the method only works on constant load demand data, the SOC of EVs at 8 AM is a mean value and on any individual day the mean may be less. Additionally, it is unclear whether the resulting policy is optimal, and some discharging in consecutive hours is present for EVs with less than 25% charge.")

        st.image('F8.png', caption='Fig. 8. Demand Prediction LSTM')
        st.write("This figure (Fig. 8.) illustrates the demand prediction capability of the LSTM component in the LSTM-ILP model. It compares the actual demand profile (likely a combination of grid foundation load, EV demand, and discharge capacity) with the demand predicted by the LSTM neural network. This comparison helps evaluate the accuracy of the demand forecasting, which is crucial for the subsequent optimization step.")

        st.write("The LSTM-ILP strategy experiences daily changes in both total load demand and charging prices due to the unpredictability of electric vehicle driving behaviour. We determined the range of variation in grid load differences, charging costs, and aggregator income using V2G dispatching. The average grid load difference, with a 95% confidence range of [395.7, 454.6], was determined after 50 runs. Even at the top limit, this technique efficiently lowers load disparities. Effective cost reduction is demonstrated by the average EV charging cost of 1980.1 yuan, with a 95% confidence interval of [1696.3, 2263.9]. With a 95% confidence interval of [1205.2, 1448.0], the aggregators' average profit of 1326.6 yuan ensures profitability even at the lowest income level.")
        
        st.subheader("5. Conclusion")
        st.write("The optimization of vehicle-to-grid (V2G) operations in residential areas with electric cars (EVs) requires the use of powerful machine learning models such as LSTM and Q-learning. These models aid in the development of EV charging and discharging schedules that lessen peak loads, increase grid stability, and save money for EV owners.")
        st.write("The LSTM model places more emphasis on discharging EVs to support the grid during times of peak demand than it does on charging EVs in the evening and early morning. This tactic aids in cost-effectiveness and energy-usage optimization.")
        st.write("In order to maximize grid support capacity during periods of low demand and take advantage of cheaper power rates, the Q-learning model arranges EV charging during off-peak hours.")
        st.write("These various methods demonstrate how machine learning techniques may be used to adapt V2G tactics to changing grid circumstances and user preferences with flexibility and efficacy. The primary objective is to optimize charging patterns for cost savings and user convenience while reducing the burden on the grid during peak hours.")
        st.write("By encouraging the integration of renewable energy sources, lowering greenhouse gas emissions, and constructing a more robust energy infrastructure, putting these optimized schedules into practice helps create a more sustainable energy environment.")
        st.write("Using DQN and LSTM models to create customized V2G plans is a big step toward building a more intelligent and flexible energy system that will allow EVs to be flexible assets throughout the shift to sustainable energy.")
        st.write("Future topics for study can include investigating hybrid techniques that incorporate the best features of both approaches, extending the optimization framework to include energy storage systems and renewable energy sources, and taking into account more complicated situations with a variety of EV kinds and user preferences.")
        
        st.subheader("6. REFERENCES")
        st.write("[1] Yanyu Zhang, Xinpeng Rao, Chunyang Liu, Xibeng Zhang, Yi Zhou, A cooperative EV charging scheduling strategy based on double deep Q-network and Prioritized experience replay, Engineering Applications of Artificial Intelligence, 2023")
        st.write("[2] Lina Ren, Mingming Yuan, Xiaohong Jiao, Electric vehicle charging and discharging scheduling strategy based on dynamic electricity price, Engineering Applications of Artificial Intelligence,2023")
        st.write("[3] Schaul, T. Quan, J., Antonoglou, I., 2015. Prioritized experience replay. arXiv preprint arXiv:1511.05952.")
        st.write("[4] Volodymyr Mnih and Koray Kavukcuoglu and David Silver and Alex Graves and Ioannis Antonoglou and Daan Wierstra and Martin Riedmiller,Playing Atari with Deep Reinforcement Learning, 2013, arXiv.")
        st.write("[5] Zhang, F., Yang, Q., An, D., 2021b. CDDPG: A deepreinforcementlearning-based approach for electric vehicle charging control. IEEE Internet Things J. 8 (5),3075–3087.")
        st.write("[6] Kyungbae Shin, Jungwoo Shin, Kyung-Ok Cho, Reinforcement learning based EV charging optimization considering intermittent renewable energy, Engineering Applications of Artificial Intelligence - 2017")
        st.write("[7] Zuting Huang, Yu Zhang, Chuanwen Jiang, Yang Li, Intelligent Charging Strategy for Electric Vehicles Based on Q-Learning, IEEE Access2018")
        st.write("[8] Shuai Li, Yanan Li, Jie Zhang, Lijun Zhang, Jingjing Zhang, Wenjuan Zhang, Shuai Li, Ruixi Zhang, Distributed Transformer Joint Optimization Method using Multi-Agent Deep Reinforcement Learning for Electric Vehicle Charging. IEEE Transactions on Industrial Informatics - 2021.")
        st.write("[9] Felix Tuchnitz, Niklas Ebell, Jonas Schlund, Marco Pruckner Development and Evaluation of a Smart Charging Strategy for an Electric Vehicle Fleet Based on Reinforcement Learning. Applied Energy, ISSN: 0306- 2619, 2021. http://dx.doi.org/10.1016/j.apenergy.2020.116382")
        st.write("[10] Da Silva, F.L., Nishida, C.E., Roijers, D.M. and Costa, A.H.R., 2019. Coordination of electric vehicle charging through multiagent reinforcement learning. IEEE Transactions on Smart Grid, 11(3), pp.2347-2356.")
        st.write("[11] Hao, X., Chen, Y., Wang, H., Wang, H., Meng, Y. and Gu, Q., 2023. A V2G-oriented")








if __name__ == "__main__":
    main()
