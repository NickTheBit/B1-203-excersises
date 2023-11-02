# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import definitions as d
import random as rd

# Q(time,tankLevel,pump enabled)
QTable = np.ones((36, 4, 2), dtype=int)


def startup():
	# Initializing environment
	env = d.Enviro(150, 3)

	# Learning parameters
	epsilon = 0.8
	learning_rate = 0.5
	discount_factor = 0.3

	# Mapping out the tank
	tankState = env.getTankLevelDiscrete()

	# Looping over every state, lol
	for i in range(0, len(QTable) - 1, 1):
		# This is the espilon-greedy algo.
		if (rd.random() > epsilon):
			# Optimal sellection based on Q
			currentAction = max(QTable[i][tankState])
		else:
			currentAction = rd.randint(0, 1)

		# Updating the Q - Value
		QTable[i][tankState][currentAction] = QTable[i][tankState][currentAction] + \
			learning_rate*(-env.cost(tankState, currentAction)+discount_factor*QTable[i+1][tankState][max(QTable[i+1][tankState+1])]-QTable[i][tankState][currentAction])
	
		print(QTable)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	startup()
