# Here we define all the functions needed for q-learning to function

class supportFunctions:
	# Internally keeping the current tank level before discretization.
	currentTankLevel = 0

	# Tank limits for cost function.
	lowerLimit=2
	upperLimit=4
	
	def __init__(self, newTankMaxLevel, newTankDescreteLevels) -> None:
		self.tankMaxLevel = newTankMaxLevel
		self.tankDircreteLevels = newTankDescreteLevels
		self.singleLevelQuantiny = self.tankMaxLevel / self.tankDircreteLevels

	def getTankLevelDiscrete(self, newCurrentTankLevel):
		self.currentTankLevel = newCurrentTankLevel
		# Compute and map out tank level to discrete states.
		retVal = int(self.currentTankLevel / self.singleLevelQuantiny)-1

		# Since no soft limit, the tank will go negative, this patches an underlying problemo
		# Coding drunk is extremely fun and I highly recomend it.
		if (retVal < 0):
			retVal = 0

		return retVal
	
	def cost(self, flow):
		# Discrete cost
		barrierCost = 0
		if self.currentTankLevel >= self.upperLimit:
			barrierCost =(self.currentTankLevel-self.upperLimit)**2
		elif self.currentTankLevel <= self.lowerLimit:
			barrierCost = (self.lowerLimit-self.currentTankLevel)**2
		else:
			barrierCost = 0

		# This accounted for cost of operation, but I don't give a shit right now
		return flow+barrierCost
