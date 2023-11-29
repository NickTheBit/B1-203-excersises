# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import definitions as d
import random as rd
import matplotlib.pyplot as plt
import pandas as pd 
import copy

# Q(time,tankLevel,pump enabled)
#QTable= np.random.randint(low=0, high=5, size=(24, 19,2))
QTable = np.zeros((24, 19, 2))
LevelHistory=[]
ActionHistory=[]
N_days=2*500 # Number of days that we are running the simulation


def startup():
	# Initializing environment
	global prev_QTable
	prev_QTable=0
	never_entered=1
	env = d.Enviro(1, 20,1) # arguments: initial tank level, pump status, noise status (either 0 or 1)

	# Learning parameters
	learning_rate = 0.2
	discount_factor = 0.4
	x_intersect=2*400 # day at which epsilon decays to 0
	a=9/x_intersect
	J=0
	

	for j in range(0,N_days+1):
		# Looping over every state, lol
		#learning_rate = 5/(j+1)
		for i in range(0, 23+1):
			# Mapping out the tank
			tankState = env.getTankLevelDiscreteVariable()
			epsilon=1-np.log10(a*j+1) # logaritmic decay
			optimalAction=np.argmin(QTable[i][tankState])
			# This is the espilon-greedy algo.
			if (rd.random() > epsilon):
				# Optimal sellection based on Q
				currentAction = optimalAction
				
			else:
				currentAction = rd.randint(0, 1)
			
		
			# Take action a and observe s',r'
			env.updateWaterLevel(i, currentAction*env.pumpFlowRate)
			futureTankState= env.getTankLevelDiscreteVariable()  

	
			#We penalize more if we get out of
			Jnext=env.cost(currentAction)
		
			

			# Updating the Q - Value
			if i <23:
				QTable[i,tankState,currentAction] =(1-learning_rate)*QTable[i, tankState, currentAction]+learning_rate*(Jnext+discount_factor*QTable[i+1,futureTankState, np.argmin(QTable[i+1,futureTankState])]-QTable[i,tankState,currentAction])
			else:
				QTable[i, tankState, currentAction] = (1-learning_rate)*QTable[i,tankState,currentAction]+learning_rate*(Jnext+discount_factor*QTable[0,futureTankState, np.argmin(QTable[0,futureTankState])]-QTable[i,tankState,currentAction])

			ActionHistory.append(currentAction*env.pumpFlowRate)
			LevelHistory.append(env.currentTankLevel)

			
			if j==N_days:
				J=abs(J+Jnext)
				print(QTable)

			# Check for convergence
			difference = QTable - prev_QTable
			absolute_error = np.abs(difference)
			relative_error = absolute_error / (np.abs(QTable) + 1e-10)
		
			matrix_np = np.array(relative_error)
			threshold=0.0001

			prev_QTable = copy.copy(QTable)
		
			if(np.all((matrix_np < threshold)) and never_entered and j>0):
				never_entered=0
				convergenceDay=j
				convergenceHour=i
				print(f"Convergence day={convergenceDay}, hour={convergenceHour}")

		
		
	print(f"G={J}")
	

	#To save the last QTable
	# Convert the array to a DataFrame
	#df = pd.DataFrame(np.max(QTable, axis=2))
	df_action_0=pd.DataFrame(QTable[:,:,0])
	df_action_1=pd.DataFrame(QTable[:,:,1])

    # Save the DataFrame to a CSV file
	df_action_0.to_csv('df_action_0.csv', index=False, header=False)
	df_action_1.to_csv('df_action_1.csv', index=False, header=False)
	
	

			




		
		

	


	
			
		

	plt.style.use('dark_background')

	plt.figure(1)
	plt.suptitle("Water Level and Action over time")  # Title for the entire figure

	# First subplot
	plt.subplot(2, 1, 1)
	plt.plot(LevelHistory)
	plt.ylabel("Water Level [L]")  # Y-axis label for the first subplot
	plt.title("Water Level over time [L]")  # Title for the first subplot

	# Second subplot
	plt.subplot(2, 1, 2)
	plt.plot(ActionHistory)
	plt.ylabel("Action")  # Y-axis label for the second subplot
	plt.xlabel("time [h]")  # X-axis label for the second subplot

	# Adjust layout
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to provide space for the suptitle


	plt.figure(2)

	# First subplot
	plt.subplot(2, 1, 1)
	plt.plot(LevelHistory[len(LevelHistory) - 24 : len(LevelHistory)], label="Levels last day")
	plt.plot(ActionHistory[len(ActionHistory) - 24 : len(ActionHistory)], label="Actions last day")
	plt.ylabel("Levels/Actions")  # Y-axis label
	plt.title("Water Level and Actions over the last day [L]")  # Title for the first subplot
	plt.legend()  # Adding a legend to distinguish between LevelHistory and ActionHistory

	# Second subplot
	plt.subplot(2, 1, 2)
	plt.stem(env.consumption_record[len(env.consumption_record) - 24 : len(env.consumption_record)])
	plt.xlabel("Time [h]")  # X-axis label
	plt.ylabel("Demand")  # Y-axis label
	plt.title("Demand vs Water Level over the last day [L]")  # Title for the second subplot


	# Adjust layout to prevent overlap
	plt.tight_layout()
	plt.show()

	

	
	
	
	

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	startup()
