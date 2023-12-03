# Packages loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import WaterNetwork as WN
import SupportFunctions as sf
import random as rd

# Visualization packages
from rich.progress import track


### Class defintions ####################################################################################################

class PumpingStationControl:
    # Setup Q-Learning parameters.
    descreteStates = 7

    def __init__(self, total_iterrations):
        # Parameters
        self.hmin = 2.4  # [m] min level in the tank
        self.hmax = 3.2  # [m] max level in the tank
        self.num_of_pumps = 3  # Number of pumps in the pumping station (WHY WAS THIS A FLOAT???? DO YOU HAVE HALF PUMPS IN YOUR VILLAGE???)
        # State initialization
        self.num_running_pumps = 0.0  # Number of running pumps
        self.speed = 1.0  # [0-1] Speed of active pumps, I'll politely ignore that for now.

        # Normally we are not aware of the total steps of our array, this is CHEATING
        self.total_iterations = total_iterrations + 1

        # For this iteration the controller makes a decision once per hour.
        self.Qtable = np.ones((25, self.descreteStates, self.num_of_pumps + 1))
        # epsilon parameter
        self.epsilon = 0.1  # this should be adjustable
        self.learning_rate = 0.07  # this should also do something.
        self.discount_factor = 0.8  # This is probably too high

        # Starting action
        self.currentAction = 1

        # Initializing Q-learning
        self.ql = sf.SupportFunctions(10, self.descreteStates, self.learning_rate, self.discount_factor)

    def sim_step(self, water_level, current_time, step, total_steps):
        # We are updating the Q-values, unless the step = 0

        # Updating controller data
        tank_current_state = self.ql.getTankLevelDiscrete(water_level)

        # Take action a and observe s',r'
        self.num_running_pumps = self.currentAction
        tank_state = self.ql.getTankLevelDiscrete(water_level)

        # We penalize more if we get out of bounds
        reward = self.ql.reward(water_level)

        current_q_value = self.Qtable[current_time][tank_current_state][self.currentAction]
        # Updating the Q - Value
        self.Qtable[current_time][tank_current_state][self.currentAction] = self.ql.compute_q(current_q_value, reward,
                                                                                            np.argmax(self.Qtable[
                                                                                                          current_time + 1][
                                                                                                          tank_state]))

        # Selecting next step
        optimalAction = np.argmax(self.Qtable[current_time][tank_state])

        # Setting epsilon depending on simulation progress
        # If statement is to avoid division by zero
        if step != 0:
            # self.epsilon = (step / total_steps)
            self.epsilon = np.log10(10 - (step / total_steps) * 10)

        # This is the epsilon-greedy algo.
        if rd.random() > self.epsilon:
            # Optimal selection based on Q
            self.currentAction = optimalAction
        else:
            self.currentAction = rd.randint(0,
                                            self.num_of_pumps)  # The -1 is because apparently it includes the number in the random sellection.

    def get_outputs(self):
        return self.speed, self.num_running_pumps


### Simulation parameters ##################################################################################################
print("Initializing the simulation.")

# Simulation settings
simTime = 40.0 * 24.0 * 3600.0

# Setup instance of the simulation
waterNetwork = WN.waterSupplyNetworkObject()
simSteps = int(simTime / waterNetwork.getSampTime())
controller = PumpingStationControl(simSteps)

### Simulation ############################################################################################################
print("Setting up memory.")
time = np.zeros(simSteps)
level = np.zeros(simSteps)
demand1 = np.zeros(simSteps)
demand2 = np.zeros(simSteps)
pump_station_flow = np.zeros(simSteps)
pump_station_pressure = np.zeros(simSteps)
pump_station_power = np.zeros(simSteps)
pump_speed = np.zeros(simSteps)
num_of_running_pumps = np.zeros(simSteps)

# Each step has a duration of 15 mins.
for k in track(range(simSteps), description="Simulation running..."):

    # Water supply networkx
    waterNetwork.simStep(pump_speed[k], num_of_running_pumps[k])
    time[k], level[k], pump_station_flow[k], pump_station_pressure[k], pump_station_power[k], demand1[k], demand2[
        k] = waterNetwork.getMeasurements()

    # Controller is only engaged every 60 minutes
    if k % 4 == 0:
        controller.sim_step(level[k], int(k / 4) % 24, k, simSteps)

    if k < simSteps - 1:
        pump_speed[k + 1], num_of_running_pumps[k + 1] = controller.get_outputs()

### Plot results ##########################################################################################################
fig, axs = plt.subplots(5, 1)
axs[0].plot(time, level, label='Tank Level')
axs[0].legend()
axs[0].set_ylabel("level [m]")
axs[1].plot(time, pump_station_flow, label='pump flow')
axs[1].plot(time, demand1, label='demand flow 1')
axs[1].plot(time, demand2, label='demand flow 2')
axs[1].legend()
axs[1].set_ylabel("flow [m3/h]")
axs[2].plot(time, num_of_running_pumps, label='running pumps')
axs[2].legend()
axs[2].set_ylabel('number [-]')
axs[3].plot(time, pump_speed, label='pump speed')
axs[3].legend()
axs[3].set_ylabel("speed [0-1]")
axs[4].plot(time, pump_station_power, label='pump speed')
axs[4].legend()
axs[4].set_ylabel("power [KW]")
axs[4].set_xlabel("time [sec]")

plt.show()
