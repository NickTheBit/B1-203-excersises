# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import definitions as d
import random as rd
import matplotlib.pyplot as plt
import pandas as pd
# Q(time,tankLevel,pump enabled)
QTable = np.zeros((24, 19, 2))
LevelHistory=[]
ActionHistory=[]
w = np.ones((24 , 19, 2))
N_days=1000 # Number of days that we are running the simulation



def startup():
	# Initializing environment
	env = d.Enviro(8, 20,0) # arguments: initial tank level, pump status, noise status (either 0 or 1)

	# Learning parameters
	learning_rate = 0.1
	discount_factor = 0.8
	x_intersect=500 # day at which epsilon decays to 0
	a=9/x_intersect

	for j in range(0,N_days+1):
		# Looping over every state, lol
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
			if currentAction==1 and futureTankState==8:
				Jnext=env.cost(currentAction)*5
			elif currentAction==0 and futureTankState==0:
				Jnext=env.cost(currentAction)*5
			else:
				Jnext=env.cost(currentAction)*5
				
			

			# Updating the Q - Value
			if i <=22:
				QTable[i][tankState][currentAction] =(1-learning_rate)*QTable[i][tankState][currentAction]+learning_rate*(Jnext+discount_factor*QTable[i+1][futureTankState][np.argmin(QTable[i+1][futureTankState])]-QTable[i][tankState][currentAction])
				RBFerror = env.RBFerr(QTable[i][tankState][currentAction], w[i][tankState][currentAction], i)
				w[i][tankState][currentAction] = env.sgd_update(w[i][tankState][currentAction], 0.1, RBFerror, 1)
				

			else:
				QTable[i][tankState][currentAction] =(1-learning_rate)*QTable[i][tankState][currentAction]+learning_rate*(Jnext+discount_factor*QTable[0][futureTankState][np.argmin(QTable[0][futureTankState])]-QTable[i][tankState][currentAction])
				RBFerror = env.RBFerr(QTable[i][tankState][currentAction], w[i][tankState][currentAction], i)
				w[i][tankState][currentAction] = env.sgd_update(w[i][tankState][currentAction], 0.1, RBFerror, 1)
			
			ActionHistory.append(currentAction*env.pumpFlowRate)
			LevelHistory.append(env.currentTankLevel)
	print(len(LevelHistory))
	RDFweights0=pd.DataFrame(w[:,:,0])
	RDFweights1=pd.DataFrame(w[:,:,1])


    # Save the DataFrame to a CSV file
	RDFweights0.to_csv('RDFweights0.csv', index=False, header=False)
	RDFweights1.to_csv('RDFweights1.csv', index=False, header=False)		
	

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
	waterheight = np.linspace(env.tankMinLevel, env.tankMaxLevel, 800)
	RBFQ0 = []
	RBFQ1 = []
	for g in range(24):
		RBFQ0.append(env.RBFapprox(w[g,:,0], waterheight))
	for n in range(24):
		RBFQ1.append(env.RBFapprox(w[n,:,1], waterheight))
	
	plt.figure(3)
	line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12 = plt.plot(waterheight, RBFQ0[0],waterheight, RBFQ0[2],waterheight, RBFQ0[4],waterheight, RBFQ0[6],waterheight, RBFQ0[8],waterheight, RBFQ0[10],waterheight, RBFQ0[12],waterheight, RBFQ0[14],waterheight, RBFQ0[16],waterheight, RBFQ0[18],waterheight, RBFQ0[20], waterheight, RBFQ0[22])
	plt.xlabel("Water Level")  # X-axis label
	plt.ylabel("Q")  # Y-axis label
	plt.figlegend([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12], ['0h', '2h','4h','6h','8h','10h','12h','14h', '16h', '18h', '20h', '22h'])
	plt.title("Q vs Water Level action 0")  # Title for the second subplot
	plt.figure(4)
	line112, line21, line31, line41, line51, line61, line71, line81, line91, line101, line111, line121 = plt.plot(waterheight, RBFQ1[0],waterheight, RBFQ1[2],waterheight, RBFQ1[4],waterheight, RBFQ1[6],waterheight, RBFQ1[8],waterheight, RBFQ1[10],waterheight, RBFQ1[12],waterheight, RBFQ1[14],waterheight, RBFQ1[16],waterheight, RBFQ1[18],waterheight, RBFQ1[20], waterheight, RBFQ1[22])
	plt.xlabel("Water Level")  # X-axis label
	plt.ylabel("Q")  # Y-axis label
	plt.figlegend([line112, line21, line31, line41, line51, line61, line71, line81, line91, line101, line111, line121], ['0h', '2h','4h','6h','8h','10h','12h','14h', '16h', '18h', '20h', '22h'])
	plt.title("Q vs Water Level action 1")  # Title for the second subplot	
	plt.show()

	
	
	
	

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	startup()
