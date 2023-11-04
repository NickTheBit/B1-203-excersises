# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import definitions as d
import random as rd
import matplotlib.pyplot as plt

# Q(time,tankLevel,pump enabled)
QTable = np.ones((24, 8, 2))
LevelHistory=[]
ActionHistory=[]
N_days=1000 # Number of days that we are running the simulation

def startup():
	# Initializing environment
	env = d.Enviro(1, 10)

	# Learning parameters
	learning_rate = 0.9
	discount_factor = 1

	for j in range(0,N_days):
		# Looping over every state, lol
		for i in range(0, 23):
			# Mapping out the tank
			tankState = env.getTankLevelDiscrete()
			epsilon=1-0.4*np.log10(j+1)
			optimalAction=np.argmin(QTable[i][tankState])
			# This is the espilon-greedy algo.
			if (rd.random() > epsilon):
				# Optimal sellection based on Q
				currentAction = optimalAction
				
			else:
				currentAction = rd.randint(0, 1)
			
		
			# Take action a and observe s',r'
			
			env.updateWaterLevel(i, i+1, currentAction*env.pumpFlowRate)
			futureTankState= env.getTankLevelDiscrete()  
			Jnext=env.cost(currentAction)

			# Updating the Q - Value
			QTable[i][tankState][currentAction] = QTable[i][tankState][currentAction]+learning_rate*(Jnext+discount_factor*QTable[i+1][futureTankState][np.argmin(QTable[i+1][futureTankState])]-QTable[i][tankState][currentAction])
			ActionHistory.append(currentAction*env.pumpFlowRate)
			LevelHistory.append(env.currentTankLevel)
	print(QTable)
	
			
		
	plt.style.use('dark_background') 
	plt.figure(1)
	plt.title("Water Level over time")
	plt.xlabel("time[h]")
	plt.ylabel("[L]")
	plt.plot(LevelHistory)	
	plt.plot(ActionHistory)		
	
	plt.figure(2)
	plt.title("Water Level over the last day")
	plt.xlabel("time[h]")
	plt.ylabel("[L]")
	plt.plot(LevelHistory[len(LevelHistory)-24:len(LevelHistory)])
	plt.plot(ActionHistory[len(ActionHistory)-24:len(ActionHistory)])
	
	plt.show()

	
	
	
	

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	startup()
