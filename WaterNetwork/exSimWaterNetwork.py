# Packages loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import WaterNetwork as WN
import supportFunctions as sf
import random as rd

### Class defintions ####################################################################################################

class pumpingStationControl:
	# Setup Q-Learning parameters.
	descreteStates = 5

	def __init__(self, cheating_Total_Iterrations):
		# Parameters
		self.hmin = 2.4                # [m] min level in the tank
		self.hmax = 3.2                # [m] max level in the tank
		self.num_of_pumps = 3          # Number of pumps in the pumping station (WHY WAS THIS A FLOAT???? DO YOU HAVE HALF PUMPS IN YOUR VILLAGE???)
		# State initalization
		self.num_running_pumps = 0.0   # Number of running pumps
		self.speed = 1.0   # [0-1] Speed of active pumps, I'll politely ignore that for now.

		# Normally we are not aware of the total steps of our array, this is CHEATING
		self.totalIterrations = cheating_Total_Iterrations + 1

		# For this itteration the controller makes a decision once per hour.
		self.Qtable = np.ones((24, self.descreteStates, self.num_of_pumps + 1))
		# epsilon parameter
		self.epsilon = 0.7  # this approach sucks
		self.learning_rate = 0.3  # this should also do something.
		self.discount_factor = 1 # This is probably too high

		# Starting action
		self.currentAction = 1

		# Initializing Q-learning
		self.ql = sf.supportFunctions(3.2, self.descreteStates, self.learning_rate, self.discount_factor)

	def simStep(self, level, time):
		# We are updating the Q-values, unless the step = 0

		# Take action a and observe s',r'
		tankState = self.ql.getTankLevelDiscrete(level)

		# We penalize more if we get out of bounds
		reward = self.ql.reward(tankState)


		current_Q_value = self.Qtable[time][tankState][self.currentAction]
		# Updating the Q - Value
		self.Qtable[time][tankState][self.currentAction] = self.ql.compute_q(current_Q_value, reward, np.argmax(self.Qtable[time+1][tankState]))
		
		# Sellecting next step
		optimalAction = np.argmax(self.Qtable[time][tankState])
		# This is the espilon-greedy algo.
		if (rd.random() > self.epsilon):
			# Optimal sellection based on Q
			self.currentAction = optimalAction
		else:
			self.currentAction = rd.randint(0, self.num_of_pumps) # The -1 is because apparently it includes the number in the random sellection.

		# Updating controller data
		self.num_running_pumps = self.currentAction

	############### Old Sim-step ##############################
	# Dh = (self.hmax - self.hmin)/self.num_of_pumps
	# if level < Dh*(self.num_of_pumps - self.num_running_pumps - 1.0) + self.hmin :
	#     self.num_running_pumps = self.num_running_pumps + 1.0
	# if level > Dh*(self.num_of_pumps - self.num_running_pumps + 1.0) + self.hmin :
	#     self.num_running_pumps = self.num_running_pumps - 1.0
	# self.speed = 1

	def getOutputs(self):
		return self.speed, self.num_running_pumps


### Simulation parameters ##################################################################################################
print("Initialization of the simulation")

# Simulation settings
simTime = 40.0*24.0*3600.0

# Setup instance of the simulation
waterNetwork = WN.waterSupplyNetworkObject()
simSteps = int(simTime/waterNetwork.getSampTime())
controller = pumpingStationControl(simSteps)

### Simulation ############################################################################################################
print("Run simulation")
time = np.zeros(simSteps)
level = np.zeros(simSteps)
demand1 = np.zeros(simSteps)
demand2 = np.zeros(simSteps)
pump_station_flow = np.zeros(simSteps)
pump_station_pressure = np.zeros(simSteps)
pump_station_power = np.zeros(simSteps)
pump_speed = np.zeros(simSteps)
num_of_running_pumps = np.zeros(simSteps)

for k in range(simSteps):

	# Water supply networkx
	waterNetwork.simStep(pump_speed[k], num_of_running_pumps[k])
	time[k], level[k], pump_station_flow[k], pump_station_pressure[k], pump_station_power[k], demand1[k], demand2[k] = waterNetwork.getMeasurements()

	# Controller
	controller.simStep(level[k], simSteps % 24)
	if k < simSteps-1:
		pump_speed[k+1], num_of_running_pumps[k+1] = controller.getOutputs()


### Plot results ##########################################################################################################
print("Plot results")
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

