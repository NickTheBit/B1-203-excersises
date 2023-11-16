# Here we define all the functions needed for q-learning to function

class supportFunctions:
	# Tank limits for cost function.
	lowerLimit=2
	upperLimit=4
	
	def __init__(self, newTankMaxLevel, newTankDescreteLevels, new_learning_rate, new_discount_factor) -> None:
		self.tankMaxLevel = newTankMaxLevel
		self.tankDircreteLevels = newTankDescreteLevels
		self.singleLevelQuantiny = self.tankMaxLevel / self.tankDircreteLevels
		self.learning_rate = new_learning_rate
		self.discount_factor = new_discount_factor

	def getTankLevelDiscrete(self, newCurrentTankLevel):
		self.currentTankLevel = newCurrentTankLevel
		# Compute and map out tank level to discrete states.
		retVal = int(self.currentTankLevel / self.singleLevelQuantiny)-1

		# Since no soft limit, the tank will go negative, this patches an underlying problemo
		# Coding drunk is extremely fun and I highly recomend it.
		if (retVal < 0):
			retVal = 0

		if (retVal > self.tankMaxLevel):
			retVal = self.upperLimit

		return retVal
	
	def reward(self, tankLevelState):
		# Extreme punishment for exceeding legal limits
		reward = 0
		if tankLevelState >= self.upperLimit:
			reward = 0
		elif tankLevelState <= self.lowerLimit:
			reward = 0
		else:
			reward = 1

		# This accounted for cost of operation, but I don't give a shit right now
		return reward
	
	def compute_q(self, currentValue, reward, estimate_optimal_future_value):
		return (1 - self.learning_rate) * currentValue + self.learning_rate * (reward + self.discount_factor * estimate_optimal_future_value)
		
