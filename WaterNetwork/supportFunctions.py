# Here we define all the functions needed for q-learning to function

class supportFunctions:
	def __init__(self, newTankMaxLevel, newTankDescreteLevels, new_learning_rate, new_discount_factor) -> None:
		self.tankMaxLevel = newTankMaxLevel
		self.tankDircreteLevels = newTankDescreteLevels
		self.singleLevelQuantiny = self.tankMaxLevel / self.tankDircreteLevels
		self.learning_rate = new_learning_rate
		self.discount_factor = new_discount_factor

	def getTankLevelDiscrete(self, newCurrentTankLevel):
		# Compute and map out tank level to discrete states.
		retVal = int(newCurrentTankLevel / self.singleLevelQuantiny)

		# Since no soft limit, the tank will go negative, this patches an underlying problemo
		# Coding drunk is extremely fun and I highly recomend it.
		if (retVal < 0):
			retVal = 0

		if (retVal > self.tankMaxLevel):
			retVal = self.upperLimit

		return retVal


	def reward(self, current_tank_level):
		# Extreme punishment for exceeding legal limits
		if (current_tank_level > self.tankMaxLevel):
			return  - (current_tank_level - self.tankMaxLevel)**2
		elif (current_tank_level < 0):
			return  - current_tank_level **2
		else:
			return -abs(self.tankMaxLevel/2 - current_tank_level)

		# This needs to account for cost of operation, but I don't give a shit right now
		return reward
	
	def compute_q(self, currentValue, reward, estimate_optimal_future_value):
		return (1 - self.learning_rate) * currentValue + self.learning_rate * (reward + self.discount_factor * estimate_optimal_future_value)
		
